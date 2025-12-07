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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def normalize_keyword(s: str) -> str:
    s = str(s).lower().strip()
    return "".join(c for c in s if c.isalnum() or c.isspace())


st.set_page_config(
    page_title="German SEO Planner",
    layout="wide",
    page_icon="🇩🇪"
)


if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_results = None
    st.session_state.synonyms = []
    st.session_state.strategy_text = ""
    st.session_state.working_groq_model = None
    st.session_state.current_topic = ""


@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token: Optional[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hf_token:
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)

    try:
        m = SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
        return m
    except:
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)


def process_keywords_gemma(df_keywords, seeds, threshold, hf_token):
    if df_keywords is None or df_keywords.empty:
        return None

    candidates = df_keywords["German Keyword"].astype(str).tolist()
    base_seeds = [s.strip() for s in seeds if s.strip()]

    topic = st.session_state.get("current_topic", "")
    if topic:
        base_seeds.append(topic.strip())

    model = load_gemma_model(hf_token)

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


def run_groq(api_key: str, prompt: str):
    client = Groq(api_key=api_key)

    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "llama-3.1-8b-instant",
    ]

    if st.session_state.working_groq_model:
        models.insert(0, st.session_state.working_groq_model)

    last_error = None

    for m in models:
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                model=m,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            st.session_state.working_groq_model = m
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_error = str(e)
            if "401" in last_error:
                return {"error": "INVALID_KEY"}
            if "429" in last_error:
                time.sleep(2)
                continue
            continue

    return {"error": f"All models failed. Last error: {last_error}"}


def get_cultural_translation(api_key: str, keyword: str):
    prompt = f"""
    Act as a German SEO expert.

    English phrase: "{keyword}"

    Return the 3 most semantically accurate German search terms.
    JSON only:
    {{
      "synonyms": ["t1","t2","t3"],
      "explanation": "..."
    }}
    """

    out = run_groq(api_key, prompt)
    if "error" in out:
        return out

    syns = out.get("synonyms", [])
    if isinstance(syns, str):
        syns = [syns]
    syns = [s.strip() for s in syns if s.strip()]

    out["synonyms"] = syns
    out["explanation"] = str(out.get("explanation", "")).strip()
    return out


def fetch_suggestions(q: str):
    url = f"https://www.google.com/complete/search?client=chrome&q={q}&hl=de&gl=de"
    try:
        time.sleep(random.uniform(0.1, 0.3))
        r = requests.get(url, timeout=2.0)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return [str(x).strip() for x in data[1] if str(x).strip()]
    except:
        return []
    return []


def deep_mine(synonyms: List[str]) -> pd.DataFrame:
    seeds = [s.strip() for s in synonyms if s.strip()]
    if not seeds:
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    modifiers = [
        "",
        " symptome",
        " ursachen",
        " behandlung",
        " was tun",
        " therapie",
        " hausmittel",
        " bilder",
        " erfahrung",
        " test",
        " vergleich",
        " kosten",
    ]

    all_rows = []
    total = len(seeds) * len(modifiers)
    step = 0

    prog = st.progress(0, "Mining Google Autocomplete...")

    for s in seeds:
        for m in modifiers:
            step += 1
            prog.progress(min(step/total, 1.0))
            sug = fetch_suggestions(f"{s}{m}")
            for r in sug:
                all_rows.append({"German Keyword": r, "Seed": s})

    prog.empty()

    if not all_rows:
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    df = pd.DataFrame(all_rows)
    return df.drop_duplicates(subset=["German Keyword"]).reset_index(drop=True)


def translate_keyword(api_key: str, kw: str) -> Optional[str]:
    prompt = f"""
    Translate this German keyword to literal English.
    JSON only:
    {{
      "{kw}": {{
        "english": "<translation>"
      }}
    }}
    """

    r = run_groq(api_key, prompt)
    if "error" in r:
        return None

    if kw not in r:
        return None

    entry = r.get(kw, {})
    return str(entry.get("english", "-")).strip() or "-"


with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    st.markdown("---")
    threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.45, 0.05)


st.title("German SEO Planner")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic")
run_btn = st.button("Generate Keywords", type="primary")


if run_btn:
    if not keyword.strip():
        st.stop()
    if not api_key.strip():
        st.stop()
    if not hf_token.strip():
        st.stop()

    st.session_state.current_topic = keyword.strip()

    with st.spinner("Loading vector model..."):
        _ = load_gemma_model(hf_token)

    with st.spinner("Generating German synonyms..."):
        strat = get_cultural_translation(api_key, keyword)
        if "error" in strat:
            st.stop()

        syns = strat.get("synonyms", [])
        if not syns:
            st.stop()

        st.session_state.synonyms = syns
        st.session_state.strategy_text = strat.get("explanation", "")

    df_raw = deep_mine(syns)
    if df_raw.empty:
        st.stop()

    with st.spinner("Filtering keywords..."):
        df_f = process_keywords_gemma(df_raw, syns, threshold, hf_token)

    if df_f is None or df_f.empty:
        st.stop()

    with st.spinner("Translating keywords..."):
        df_f = df_f.copy()
        translations = []
        for kw in df_f["German Keyword"]:
            translations.append(translate_keyword(api_key, kw))
            time.sleep(0.2)

        df_f["English"] = translations

    st.session_state.df_results = df_f
    st.session_state.data_processed = True


if st.session_state.data_processed:
    st.success(f"Context: {st.session_state.strategy_text}")

    cols = st.columns(len(st.session_state.synonyms))
    for i, s in enumerate(st.session_state.synonyms):
        cols[i].markdown(
            f"<div class='metric-card'><div class='metric-val'>{s}</div></div>",
            unsafe_allow_html=True
        )

    df = st.session_state.df_results
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button("📥 Download CSV", csv, "keywords.csv", "text/csv")

    st.dataframe(
        df[["German Keyword", "English", "Relevance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relevance": st.column_config.ProgressColumn(
                "Score", format="%.2f", min_value=0, max_value=1
            )
        },
    )
