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
# 0. LOGGING CONFIG
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="German SEO Planner (Dynamic Intent)",
    layout="wide",
    page_icon="🇩🇪",
)

st.markdown(
    """
<style>
    :root {
        --primary-color: #f55036;
        --background-color: #ffffff;
        --text-color: #24292e;
    }
    .stApp {
        background-color: #ffffff;
        color: #24292e;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f55036;
        margin-bottom: 5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #f6f8fa;
        border-right: 1px solid #d0d7de;
    }
    .stTextInput input {
        background-color: #ffffff !important;
        border: 1px solid #d0d7de !important;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-ok {
        background-color: #dafbe1;
        color: #1a7f37;
    }
    .status-err {
        background-color: #ffebe9;
        color: #cf222e;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SESSION STATE
# -----------------------------------------------------------------------------
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_results = None
    st.session_state.synonyms: List[str] = []
    st.session_state.strategy_text: str = ""
    st.session_state.working_groq_model: Optional[str] = None

# -----------------------------------------------------------------------------
# 3. VECTOR ENGINE (Google Gemma via HuggingFace)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token: Optional[str]) -> SentenceTransformer:
    """
    Load the primary embedding model (Gemma) with Hugging Face auth token.
    Fallback to a smaller, commonly available model if anything fails.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hf_token:
        logger.warning(
            "No Hugging Face token provided. Falling back to all-MiniLM-L6-v2."
        )
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)

    try:
        # Assumes this model ID and token argument are valid in your environment.
        model = SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(
            device
        )
        logger.info("Loaded Gemma embedding model on %s.", device)
        return model
    except Exception as exc:
        logger.warning(
            "Failed to load Gemma model. Falling back to all-MiniLM-L6-v2. Error: %r",
            exc,
        )
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)


def process_keywords_gemma(
    df_keywords: pd.DataFrame,
    seeds: List[str],
    threshold: float,
    hf_token: Optional[str],
) -> Optional[pd.DataFrame]:
    """
    Compute semantic similarity between seed keywords and candidate keywords.
    Filter candidates by given relevance threshold.
    """
    if df_keywords is None or df_keywords.empty:
        logger.warning("process_keywords_gemma called with empty DataFrame.")
        return None

    if "German Keyword" not in df_keywords.columns:
        logger.warning(
            "DataFrame missing required 'German Keyword' column. Columns: %s",
            df_keywords.columns.tolist(),
        )
        return None

    model = load_gemma_model(hf_token)

    candidates: List[str] = (
        df_keywords["German Keyword"].astype(str).fillna("").tolist()
    )
    seeds_clean: List[str] = [s.strip() for s in seeds if s and s.strip()]

    if not seeds_clean:
        logger.warning("No valid seeds provided to process_keywords_gemma.")
        return None

    try:
        seed_vecs = model.encode(
            seeds_clean,
            prompt_name="STS",
            normalize_embeddings=True,
        )
        candidate_vecs = model.encode(
            candidates,
            prompt_name="STS",
            normalize_embeddings=True,
        )
    except TypeError:
        # For models that do not support prompt_name
        seed_vecs = model.encode(seeds_clean, normalize_embeddings=True)
        candidate_vecs = model.encode(candidates, normalize_embeddings=True)

    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    max_scores, _ = torch.max(scores_matrix, dim=1)

    df_keywords = df_keywords.copy()
    df_keywords["Relevance"] = max_scores.cpu().numpy()

    df_filtered = df_keywords[df_keywords["Relevance"] >= threshold].sort_values(
        "Relevance", ascending=False
    )

    if df_filtered.empty:
        logger.info("No keywords met the relevance threshold %.3f.", threshold)

    return df_filtered if not df_filtered.empty else None


# -----------------------------------------------------------------------------
# 4. GENERATIVE ENGINE (GROQ - SELF HEALING)
# -----------------------------------------------------------------------------
def run_groq(api_key: str, prompt: str) -> Dict[str, Any]:
    """
    Call Groq chat completion with a prioritized list of models.
    Keeps track of a working model in session state and gracefully
    handles common HTTP errors.
    """
    client = Groq(api_key=api_key)

    # Priority list (newest first)
    candidates: List[str] = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "llama-3.1-8b-instant",
    ]

    # Prefer the last working model, if any
    if st.session_state.working_groq_model:
        candidates.insert(0, st.session_state.working_groq_model)

    # Deduplicate while preserving order
    seen = set()
    ordered_candidates: List[str] = []
    for m in candidates:
        if m not in seen:
            seen.add(m)
            ordered_candidates.append(m)

    last_error: Optional[str] = None

    for model_name in ordered_candidates:
        try:
            logger.info("Trying Groq model: %s", model_name)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful SEO assistant. Output strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            st.session_state.working_groq_model = model_name
            content = chat_completion.choices[0].message.content
            return json.loads(content)
        except Exception as exc:
            last_error = repr(exc)
            logger.warning(
                "Groq model %s failed with error: %s", model_name, last_error
            )

            err_str = str(exc)

            # Model not found / bad request / decommissioned: move on to next model
            if "404" in err_str or "400" in err_str or "decommissioned" in err_str:
                continue

            # Rate limited: one retry, then continue
            if "429" in err_str:
                time.sleep(2.0)
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful SEO assistant. Output strict JSON only.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    st.session_state.working_groq_model = model_name
                    content = chat_completion.choices[0].message.content
                    return json.loads(content)
                except Exception as exc_retry:
                    logger.warning(
                        "Groq retry for model %s failed: %r", model_name, exc_retry
                    )
                    continue

            # Invalid key: do not keep trying further models
            if "401" in err_str:
                return {"error": "INVALID_KEY"}

            # For any other error, move on to next model
            continue

    return {"error": f"All Groq models failed. Last error: {last_error}"}


def get_cultural_translation(api_key: str, keyword: str) -> Dict[str, Any]:
    """
    Ask Groq to produce German cultural synonyms and a brief explanation.
    Normalizes output structure for downstream use.
    """
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term.
    2. The formal/medical term.
    3. A popular synonym.
    Output JSON format: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Brief reasoning" }}
    """

    res = run_groq(api_key, prompt)

    if "error" in res:
        return res

    raw_synonyms = res.get("synonyms", [])
    if isinstance(raw_synonyms, str):
        raw_synonyms = [raw_synonyms]

    if not isinstance(raw_synonyms, list):
        raw_synonyms = []

    synonyms: List[str] = [
        str(s).strip() for s in raw_synonyms if s and str(s).strip()
    ]
    explanation = str(res.get("explanation", "")).strip()

    res["synonyms"] = synonyms
    res["explanation"] = explanation

    return res


def batch_analyze(api_key: str, keywords: List[str]) -> Dict[str, Dict[str, str]]:
    """
    For each German keyword, ask Groq for:
      - English literal translation.
      - Search intent (Informational, Transactional, Commercial, Navigational).
    Returns a dict keyed by normalized German keyword.
    """
    cleaned_keywords: List[str] = [
        kw.strip() for kw in keywords if kw and str(kw).strip()
    ]
    if not cleaned_keywords:
        logger.warning("batch_analyze called with an empty keyword list.")
        return {}

    chunk_size = 20
    chunks: List[List[str]] = [
        cleaned_keywords[i : i + chunk_size]
        for i in range(0, len(cleaned_keywords), chunk_size)
    ]

    full_data: Dict[str, Dict[str, str]] = {}

    for chunk in chunks:
        prompt = f"""
        Act as a strict SEO Data Classifier.
        Input List: {json.dumps(chunk)}
        
        Task: For each German keyword, provide:
        1. "english": Literal translation.
        2. "intent": The Search Intent (choose one: Informational, Transactional, Commercial, Navigational).
        
        Rules:
        - "Transactional": User wants to buy NOW (e.g. buy, price, cheap).
        - "Commercial": User is researching options (e.g. best, vs, review).
        - "Informational": User wants answers (e.g. what is, symptoms, help, or generic nouns).
        
        Output strict JSON format: 
        {{ 
            "GermanKeyword1": {{ "english": "...", "intent": "..." }},
            "GermanKeyword2": {{ "english": "...", "intent": "..." }}
        }}
        """

        res = run_groq(api_key, prompt)

        if "error" in res:
            logger.warning(
                "batch_analyze chunk failed with error: %s", res.get("error")
            )
            time.sleep(0.5)
            continue

        # Normalize keys to lowercase/strip
        normalized_res: Dict[str, Dict[str, str]] = {}
        for k, v in res.items():
            if not isinstance(v, dict):
                continue
            key_norm = str(k).lower().strip()
            normalized_res[key_norm] = {
                "english": str(v.get("english", "-")).strip() or "-",
                "intent": str(v.get("intent", "-")).strip() or "-",
            }

        full_data.update(normalized_res)
        time.sleep(0.5)

    return full_data


# -----------------------------------------------------------------------------
# 5. MINING (ROBUST & RAW)
# -----------------------------------------------------------------------------
def fetch_suggestions(query: str, timeout: float = 2.0) -> List[str]:
    """
   Fetch Google autocomplete suggestions for a given query.
    Handles network/HTTP/JSON errors and returns a cleaned list of suggestions.
    """
    url = f"https://www.google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        time.sleep(random.uniform(0.1, 0.3))
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        # Expected structure: [query, [suggestions...], ...]
        if (
            isinstance(data, list)
            and len(data) > 1
            and isinstance(data[1], list)
        ):
            suggestions = [
                str(item).strip()
                for item in data[1]
                if item and str(item).strip()
            ]
            return suggestions

        logger.warning(
            "Unexpected autocomplete payload for query %r: %r", query, data
        )
    except requests.RequestException as exc:
        logger.warning(
            "Autocomplete request failed for query %r: %r", query, exc
        )
    except ValueError as exc:
        logger.warning(
            "Failed to decode autocomplete JSON for query %r: %r", query, exc
        )

    return []


def deep_mine(synonyms: List[str]) -> pd.DataFrame:
    """
    Use Google autocomplete to mine keyword variants for given German seed synonyms.
    Returns a de-duplicated DataFrame with columns: ['German Keyword', 'Seed'].
    """
    seeds: List[str] = [s.strip() for s in synonyms if s and str(s).strip()]

    if not seeds:
        logger.warning("deep_mine called with no valid synonyms.")
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    # Modifiers used only to trigger specific autocompletes
    modifiers: List[str] = [
        "",
        " kaufen",
        " test",
        " vergleich",
        " kosten",
        " erfahrung",
        " beste",
        " anleitung",
        " was ist",
    ]

    all_data: List[Dict[str, str]] = []
    total = len(seeds) * len(modifiers)

    if total <= 0:
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    prog = st.progress(0, "Mining Google Autocomplete...")

    step = 0
    for seed in seeds:
        for mod in modifiers:
            step += 1
            progress_fraction = min(step / total, 1.0)
            prog.progress(progress_fraction)

            query = f"{seed}{mod}"
            results = fetch_suggestions(query)

            for suggestion in results:
                all_data.append(
                    {
                        "German Keyword": suggestion,
                        "Seed": seed,
                    }
                )

    prog.empty()

    if not all_data:
        logger.info("deep_mine returned no suggestions.")
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=["German Keyword"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# 6. UI LAYOUT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a key from console.groq.com",
    )
    hf_token = st.text_input("Hugging Face Token", type="password")

    st.markdown("---")

    threshold = st.slider(
        "Relevance Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
    )

    if st.session_state.working_groq_model:
        st.success(f"Using: {st.session_state.working_groq_model}")

    try:
        import groq as _groq_lib  # noqa: F401

        st.markdown(
            '<span class="status-badge status-ok">✓ Groq Library Detected</span>',
            unsafe_allow_html=True,
        )
    except ImportError:
        st.markdown(
            '<span class="status-badge status-err">⚠ Groq Missing</span>',
            unsafe_allow_html=True,
        )

st.title("German SEO Planner 🇩🇪 (Dynamic Intent)")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input(
    "Enter English Topic", placeholder="e.g. heat rash in babies"
)
run_btn = st.button("Generate Keywords", type="primary")

# -----------------------------------------------------------------------------
# 7. MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------
if run_btn:
    # Basic input validation before doing any heavy work
    if not keyword or not keyword.strip():
        st.error("Please enter an English topic.")
    elif not api_key or not api_key.strip():
        st.error("Please provide a valid Groq API key.")
    elif not hf_token or not hf_token.strip():
        # You can allow empty HF token and rely on fallback if preferred.
        st.error("Please provide a Hugging Face token.")
    else:
        st.session_state.data_processed = False

        # 1. LOAD VECTORS
        with st.spinner("Loading Vector Model..."):
            try:
                _ = load_gemma_model(hf_token)
            except Exception as exc:
                logger.error("Failed to load embedding model: %r", exc)
                st.error(
                    "Failed to load embedding model. "
                    "Check your Hugging Face token and network connection."
                )
                st.stop()

        # 2. STRATEGY (Llama / Groq)
        with st.spinner("Analyzing Cultural Context..."):
            strategy = get_cultural_translation(api_key, keyword)

            if not strategy:
                st.error("No response from Groq for cultural translation.")
                st.stop()

            if "error" in strategy:
                err = strategy["error"]
                if err == "INVALID_KEY":
                    st.error("Invalid Groq API Key.")
                    st.stop()
                st.error(f"Groq error: {err}")
                st.stop()

            synonyms = strategy.get("synonyms", [])
            if not synonyms:
                st.error(
                    "Groq did not return any German synonyms. "
                    "Try adjusting the input topic."
                )
                st.stop()

            st.session_state.synonyms = synonyms
            st.session_state.strategy_text = strategy.get("explanation", "")

        # 3. MINE
        df = deep_mine(st.session_state.synonyms)

        if not df.empty:
            with st.spinner(f"Filtering {len(df)} keywords..."):
                df_filtered = process_keywords_gemma(
                    df,
                    st.session_state.synonyms,
                    threshold,
                    hf_token,
                )

            if df_filtered is not None and not df_filtered.empty:
                with st.spinner("AI Analyzing Translation & Intent..."):
                    top_keywords = (
                        df_filtered.head(40)["German Keyword"]
                        .astype(str)
                        .tolist()
                    )

                    analysis_map = batch_analyze(api_key, top_keywords)

                    def get_meta(kw: str, field: str) -> str:
                        clean_kw = kw.lower().strip()
                        meta = analysis_map.get(clean_kw)
                        if not meta:
                            return "-"
                        return meta.get(field, "-") or "-"

                    df_filtered = df_filtered.copy()
                    df_filtered["English"] = df_filtered[
                        "German Keyword"
                    ].apply(lambda x: get_meta(str(x), "english"))
                    df_filtered["Intent"] = df_filtered[
                        "German Keyword"
                    ].apply(lambda x: get_meta(str(x), "intent"))

                st.session_state.df_results = df_filtered
                st.session_state.data_processed = True
            else:
                st.warning("No keywords met the relevance threshold.")
        else:
            st.warning("No keywords found from autocomplete mining.")

# -----------------------------------------------------------------------------
# 8. RESULTS DISPLAY
# -----------------------------------------------------------------------------
if st.session_state.data_processed and st.session_state.df_results is not None:
    st.success(f"Context: {st.session_state.strategy_text}")

    if st.session_state.synonyms:
        cols = st.columns(len(st.session_state.synonyms))
        for i, syn in enumerate(st.session_state.synonyms):
            if i < len(cols):
                cols[i].markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-val">{syn}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    df_show = st.session_state.df_results

    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV",
        csv,
        "keywords.csv",
        "text/csv",
    )

    st.dataframe(
        df_show[["German Keyword", "English", "Intent", "Relevance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relevance": st.column_config.ProgressColumn(
                "Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            )
        },
    )
