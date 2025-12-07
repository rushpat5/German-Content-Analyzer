import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# NORMALIZATION FOR KEY LOOKUPS
# -----------------------------------------------------------------------------
def normalize_keyword(s: str) -> str:
    """
    Normalizes a keyword for stable dict lookups.
    """
    s = str(s).lower().strip()
    return "".join(c for c in s if c.isalnum() or c.isspace())

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German SEO Planner (Dynamic Intent)",
    layout="wide",
    page_icon="🇩🇪"
)

st.markdown("""
<style>
    :root { --primary-color: #f55036; --background-color: #ffffff; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-val { font-size: 1.2rem; font-weight: 700; color: #f55036; margin-bottom: 5px; }
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; }
    .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .status-ok { background-color: #dafbe1; color: #1a7f37; }
    .status-err { background-color: #ffebe9; color: #cf222e; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_results = None
    st.session_state.synonyms: List[str] = []
    st.session_state.strategy_text: str = ""
    st.session_state.working_groq_model: Optional[str] = None
    st.session_state.current_topic: str = ""

# -----------------------------------------------------------------------------
# VECTOR MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token: Optional[str]):
    """
    Load the primary embedding model, with fallback if Gemma fails.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hf_token:
        logger.warning("HF token missing. Using MiniLM fallback.")
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)

    try:
        model = SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
        logger.info("Loaded Gemma model.")
        return model
    except Exception as e:
        logger.warning("Gemma model failed, falling back. %r", e)
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)

def process_keywords_gemma(
    df_keywords: pd.DataFrame,
    seeds: List[str],
    threshold: float,
    hf_token: Optional[str]
) -> Optional[pd.DataFrame]:
    """
    Compute semantic similarity between candidate keywords and seed phrases.

    Improvement:
    - Uses both German synonyms and the original English topic as seeds,
      so relevance is measured to the ENTIRE query meaning, not only generic synonyms.
    """
    if df_keywords is None or df_keywords.empty:
        return None

    if "German Keyword" not in df_keywords.columns:
        logger.warning("DataFrame missing 'German Keyword' column.")
        return None

    model = load_gemma_model(hf_token)
    candidates = df_keywords["German Keyword"].astype(str).fillna("").tolist()
    base_seeds = [s.strip() for s in seeds if s and str(s).strip()]

    # Incorporate the original English topic as an extra seed
    topic = st.session_state.get("current_topic", "")
    if topic and str(topic).strip():
        base_seeds.append(str(topic).strip())

    if not base_seeds:
        logger.warning("No valid seeds for embedding filtering.")
        return None

    try:
        seed_vecs = model.encode(base_seeds, prompt_name="STS", normalize_embeddings=True)
        cand_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    except TypeError:
        seed_vecs = model.encode(base_seeds, normalize_embeddings=True)
        cand_vecs = model.encode(candidates, normalize_embeddings=True)

    scores = util.cos_sim(cand_vecs, seed_vecs)
    max_scores, _ = torch.max(scores, dim=1)

    df_keywords = df_keywords.copy()
    df_keywords["Relevance"] = max_scores.cpu().numpy()

    out = df_keywords[df_keywords["Relevance"] >= threshold].sort_values(
        "Relevance", ascending=False
    )
    return out if not out.empty else None

# -----------------------------------------------------------------------------
# GROQ ENGINE
# -----------------------------------------------------------------------------
def run_groq(api_key: str, prompt: str):
    """
    Wrapper around Groq chat.completions with model fallback.
    """
    client = Groq(api_key=api_key)

    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "llama-3.1-8b-instant",
    ]

    if st.session_state.working_groq_model:
        models.insert(0, st.session_state.working_groq_model)

    tried = set()
    last_error = None

    for m in models:
        if m in tried:
            continue
        tried.add(m)

        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=m,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            st.session_state.working_groq_model = m
            return json.loads(resp.choices[0].message.content)

        except Exception as e:
            last_error = str(e)
            if "401" in last_error:
                return {"error": "INVALID_KEY"}
            if "429" in last_error:
                time.sleep(2.0)
                continue
            else:
                continue

    return {"error": f"All models failed. Last error: {last_error}"}

# -----------------------------------------------------------------------------
# CULTURAL TRANSLATION (KEEP FULL MEANING OF PHRASE)
# -----------------------------------------------------------------------------
def get_cultural_translation(api_key: str, keyword: str):
    """
    Ask Groq for German terms representing the FULL meaning of the English phrase,
    not just the head noun.
    """
    prompt = f"""
    Act as a Native German SEO and language expert.

    English search phrase: "{keyword}"

    Task:
    - Identify the top 3 distinct German search terms that match the COMPLETE meaning
      of this phrase (including any qualifiers such as audience, location on the body,
      use-case, etc.), not just the core noun.
    - These should be realistic search queries or short keyphrases a German user would type
      when they want the same thing as the English phrase.

    Return STRICT JSON only:
    {{
      "synonyms": ["term1", "term2", "term3"],
      "explanation": "Brief reasoning of how these terms match the full meaning."
    }}
    """

    out = run_groq(api_key, prompt)
    if "error" in out:
        return out

    syns = out.get("synonyms", [])
    if isinstance(syns, str):
        syns = [syns]

    syns = [str(s).strip() for s in syns if s and str(s).strip()]
    out["synonyms"] = syns
    out["explanation"] = str(out.get("explanation", "")).strip()
    return out

# -----------------------------------------------------------------------------
# PER-KEYWORD LLM ANALYSIS (INTENT + TRANSLATION)
# -----------------------------------------------------------------------------
def batch_analyze(api_key: str, keywords: List[str]) -> Dict[str, Dict[str, str]]:
    """
    For each German keyword, get literal English translation and search intent.
    Uses one Groq call per keyword with strict key matching.
    """
    results: Dict[str, Dict[str, str]] = {}

    cleaned = [kw.strip() for kw in keywords if kw and str(kw).strip()]
    if not cleaned:
        return results

    def ask_single(kw: str, simple: bool = False) -> Optional[Dict[str, Any]]:
        if not simple:
            prompt = f"""
            JSON only:
            {{
              "{kw}": {{
                "english": "<literal translation>",
                "intent": "<Informational | Transactional | Commercial | Navigational>"
              }}
            }}
            """
        else:
            prompt = f"""
            JSON only:
            "{kw}": {{
                "english": "...",
                "intent": "..."
            }}
            """

        r = run_groq(api_key, prompt)
        if "error" in r:
            return None
        return r

    for kw in cleaned:
        key_norm = normalize_keyword(kw)
        if not key_norm:
            continue

        r = ask_single(kw, simple=False)
        if not r or kw not in r or not isinstance(r.get(kw), dict):
            r = ask_single(kw, simple=True)

        if not r or kw not in r or not isinstance(r.get(kw), dict):
            continue

        entry = r.get(kw, {})
        english = str(entry.get("english", "-")).strip() or "-"
        intent = str(entry.get("intent", "-")).strip() or "-"

        results[key_norm] = {"english": english, "intent": intent}
        time.sleep(0.2)

    return results

# -----------------------------------------------------------------------------
# AUTOCOMPLETE MINING
# -----------------------------------------------------------------------------
def fetch_suggestions(q: str) -> List[str]:
    url = f"https://www.google.com/complete/search?client=chrome&q={q}&hl=de&gl=de"
    try:
        time.sleep(random.uniform(0.1, 0.3))
        r = requests.get(url, timeout=2.0)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return [str(x).strip() for x in data[1] if x and str(x).strip()]
    except Exception as e:
        logger.warning("Autocomplete failed for %r: %r", q, e)
        return []
    return []

def deep_mine(synonyms: List[str]) -> pd.DataFrame:
    """
    Mine Google autocomplete around the German synonyms using topical modifiers.

    Improvement:
    - Modifiers are intent-related (causes, symptoms, treatment, etc.),
      which are generically relevant across niches and keep semantics tight.
    """
    seeds = [s.strip() for s in synonyms if s and str(s).strip()]
    if not seeds:
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    modifiers = [
        "",
        " symptome",
        " ursachen",
        " was tun",
        " behandlung",
        " therapie",
        " hausmittel",
        " bilder",
        " erfahrung",
        " test",
        " vergleich",
        " kosten",
    ]

    all_rows: List[Dict[str, str]] = []
    total = len(seeds) * len(modifiers)
    prog = st.progress(0, "Mining Google Autocomplete...")
    step = 0

    for s in seeds:
        for mod in modifiers:
            step += 1
            prog.progress(min(step / total, 1.0))
            query = f"{s}{mod}"
            sug = fetch_suggestions(query)
            for r in sug:
                all_rows.append({"German Keyword": r, "Seed": s})

    prog.empty()

    if not all_rows:
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["German Keyword"]).reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    st.markdown("---")
    threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.45, 0.05)

    try:
        import groq  # noqa: F401
        st.markdown('<span class="status-badge status-ok">✓ Groq Library Detected</span>', unsafe_allow_html=True)
    except ImportError:
        st.markdown('<span class="status-badge status-err">⚠ Groq Missing</span>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
st.title("German SEO Planner 🇩🇪 (Dynamic Intent)")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic", placeholder="e.g. heat rash in babies")
run_btn = st.button("Generate Keywords", type="primary")

# -----------------------------------------------------------------------------
# EXECUTION PIPELINE
# -----------------------------------------------------------------------------
if run_btn:
    if not keyword or not keyword.strip():
        st.error("Enter a topic.")
        st.stop()
    if not api_key or not api_key.strip():
        st.error("Enter Groq API key.")
        st.stop()
    if not hf_token or not hf_token.strip():
        st.error("Enter Hugging Face token.")
        st.stop()

    st.session_state.current_topic = keyword.strip()

    # VECTOR MODEL LOAD
    with st.spinner("Loading vector model..."):
        try:
            _ = load_gemma_model(hf_token)
        except Exception as e:
            st.error(f"Embedding load failed: {e}")
            st.stop()

    # CULTURAL CONTEXT
    with st.spinner("Analyzing cultural context..."):
        strat = get_cultural_translation(api_key, keyword)
        if "error" in strat:
            st.error(strat["error"])
            st.stop()

        syns = strat.get("synonyms", [])
        if not syns:
            st.error("No German terms returned for this phrase.")
            st.stop()

        st.session_state.synonyms = syns
        st.session_state.strategy_text = strat.get("explanation", "")

    # MINE
    df_raw = deep_mine(st.session_state.synonyms)
    if df_raw.empty:
        st.warning("No autocomplete suggestions.")
        st.stop()

    # FILTER by embeddings
    with st.spinner(f"Filtering {len(df_raw)} keywords..."):
        df_filtered = process_keywords_gemma(
            df_raw, st.session_state.synonyms, threshold, hf_token
        )

    if df_filtered is None or df_filtered.empty:
        st.warning("No keywords met the relevance threshold.")
        st.stop()

    # INTENT + TRANSLATION
    with st.spinner("Analyzing translation & intent..."):
        top = df_filtered.head(40)["German Keyword"].astype(str).tolist()
        analysis_map = batch_analyze(api_key, top)

        def get_meta(kw: str, field: str) -> str:
            k = normalize_keyword(kw)
            meta = analysis_map.get(k)
            if not meta:
                return "-"
            return meta.get(field, "-") or "-"

        df_filtered = df_filtered.copy()
        df_filtered["English"] = df_filtered["German Keyword"].apply(
            lambda x: get_meta(x, "english")
        )
        df_filtered["Intent"] = df_filtered["German Keyword"].apply(
            lambda x: get_meta(x, "intent")
        )

    st.session_state.df_results = df_filtered
    st.session_state.data_processed = True

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
if st.session_state.data_processed:
    st.success(f"Context: {st.session_state.strategy_text}")

    if st.session_state.synonyms:
        cols = st.columns(len(st.session_state.synonyms))
        for i, s in enumerate(st.session_state.synonyms):
            if i < len(cols):
                cols[i].markdown(
                    f"<div class='metric-card'><div class='metric-val'>{s}</div></div>",
                    unsafe_allow_html=True,
                )

    df_show = st.session_state.df_results
    csv = df_show.to_csv(index=False).encode("utf-8")

    st.download_button("📥 Download CSV", csv, "keywords.csv", "text/csv")

    st.dataframe(
        df_show[["German Keyword", "English", "Intent", "Relevance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relevance": st.column_config.ProgressColumn(
                "Score", format="%.2f", min_value=0.0, max_value=1.0
            )
        },
    )
