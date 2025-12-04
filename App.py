import streamlit as st
import pandas as pd
import requests
import json
import time
import google.generativeai as genai
import re
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION (Dejan Style)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German Vector Strategist", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    
    h1, h2, h3 { color: #111; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Metric Cards */
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 10px; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stDataFrame"] { border: 1px solid #e1e4e8; }
    
    .tech-note { font-size: 0.85rem; color: #57606a; background-color: #f6f8fa; border-left: 3px solid #0969da; padding: 12px; border-radius: 0 4px 4px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VECTOR ENGINE (EmbeddingGemma - German to German)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except Exception as e:
        st.error(f"Gemma Load Error: {e}. Ensure HF Token is valid.")
        return None

def filter_by_gemma(df_keywords, german_seeds, threshold, hf_token):
    """
    Filters keywords by comparing German Candidates vs German Seeds.
    """
    model = load_gemma_model(hf_token)
    if not model: return df_keywords
    
    # 1. Encode German Seeds (e.g. [Baby, Säugling, Neugeborenes])
    seed_vecs = model.encode(german_seeds, prompt_name="STS", normalize_embeddings=True)
    
    # 2. Encode German Candidates (e.g. [Neugeborenenakne, Babybett...])
    candidates = df_keywords['German Keyword'].tolist()
    candidate_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    
    # 3. Calculate Similarity Matrix (Candidates x Seeds)
    # We get a score for every candidate against EVERY seed.
    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    
    # 4. Take the MAX score for each candidate
    # If "Neugeborenenakne" matches "Neugeborenes" well (0.8) but "Baby" poorly (0.4),
    # we take 0.8 as the relevance score.
    max_scores, _ = torch.max(scores_matrix, dim=1)
    
    # 5. Assign & Filter
    df_keywords['Gemma Score'] = max_scores.numpy()
    
    df_filtered = df_keywords[df_keywords['Gemma Score'] >= threshold].copy()
    
    return df_filtered.sort_values('Gemma Score', ascending=False)

# -----------------------------------------------------------------------------
# 3. GENERATIVE ENGINE (Gemini Dynamic)
# -----------------------------------------------------------------------------

def get_best_model_name(api_key):
    genai.configure(api_key=api_key)
    try:
        all_models = list(genai.list_models())
        text_models = [m for m in all_models if 'generateContent' in m.supported_generation_methods]
        if not text_models: return "models/gemini-1.5-flash"
        for m in text_models:
            if 'flash' in m.name.lower(): return m.name
        return text_models[0].name
    except: return "models/gemini-1.5-flash"

def run_gemini(api_key, prompt):
    try:
        model_name = get_best_model_name(api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO.
    English Keyword: "{keyword}"
    Task: Provide the top 3 German search terms for this concept (Colloquial, Medical, Synonym).
    Return JSON: {{ "synonyms": ["term1", "term2", "term3"] }}
    """
    return run_gemini(api_key, prompt)

def batch_translate_to_english(api_key, german_keywords):
    if not german_keywords: return {}
    
    # Chunking to avoid token limits
    chunks = [german_keywords[i:i + 50] for i in range(0, len(german_keywords), 50)]
    full_map = {}
    
    prog = st.empty()
    for i, chunk in enumerate(chunks):
        prog.caption(f"Translating batch {i+1}/{len(chunks)}...")
        prompt = f"""
        Translate these German keywords to English. Literal & short.
        Input: {json.dumps(chunk)}
        Return JSON: {{ "German Word": "English Translation" }}
        """
        res = run_gemini(api_key, prompt)
        if res: full_map.update(res)
        time.sleep(0.5)
        
    prog.empty()
    return full_map

# -----------------------------------------------------------------------------
# 4. MINING ENGINE
# -----------------------------------------------------------------------------
def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine(synonyms):
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kaufen", " test"]
    all_data = []
    
    prog = st.progress(0, "Mining Google Germany...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            prog.progress(min(step/total, 1.0), f"Mining: {seed}{mod}...")
            
            results = fetch_suggestions(f"{seed}{mod}")
            
            for r in results:
                all_data.append({
                    "German Keyword": r, 
                    "Source Seed": seed,
                    "Modifier": mod.strip() if mod else "Head Term"
                })
            time.sleep(0.1)
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        return df.drop_duplicates(subset=['German Keyword'])
    return df

# -----------------------------------------------------------------------------
# 5. UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Config")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    
    st.markdown("""
    <a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;">🔑 Gemini Key</a> | 
    <a href="https://huggingface.co/settings/tokens" target="_blank" style="font-size:0.8rem;">🤗 HF Token</a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.50, 0.05)
    st.markdown(f"""
    <div class="tech-note">
    <b>Vector Logic (DE ↔ DE):</b>
    We calculate the distance between the Mined Keyword and the <b>German Head Terms</b>. 
    <br>• This ensures high relevance even if the keyword looks very different from the English input.
    </div>
    """, unsafe_allow_html=True)

st.title("German Vector Strategist 🇩🇪")
st.markdown("### Cross-Border Intelligence Powered by EmbeddingGemma")

keyword = st.text_input("Enter English Keyword", placeholder="e.g. newborn babies")
run_btn = st.button("Generate Strategy", type="primary")

if run_btn and keyword and api_key and hf_token:
    
    # 0. Load Model
    with st.spinner("Loading EmbeddingGemma..."):
        try: _ = load_gemma_model(hf_token)
        except: st.stop()

    # 1. Strategy
    with st.spinner("Translating Core Concepts..."):
        strategy = get_cultural_translation(api_key, keyword)
    
    if not strategy:
        st.error("Translation failed.")
        st.stop()

    synonyms = strategy.get('synonyms', [])
    
    # 2. Mine
    df = deep_mine(synonyms)
    
    if not df.empty:
        # 3. GEMMA FILTERING (German to German)
        with st.spinner(f"Gemma is analyzing {len(df)} keywords against German seeds..."):
            # We pass the list of synonyms (synonyms) instead of the English keyword
            df_filtered = filter_by_gemma(df, synonyms, threshold, hf_token)
            
            dropped = len(df) - len(df_filtered)
            st.success(f"Mining complete. Gemma removed {dropped} irrelevant keywords. Kept {len(df_filtered)}.")
            
        if df_filtered.empty:
            st.warning("No keywords passed the similarity threshold. Try lowering it.")
            st.stop()

        # 4. Back-Translation
        with st.spinner("Translating findings to English..."):
            germ_list = df_filtered['German Keyword'].tolist()
            translations = batch_translate_to_english(api_key, germ_list)
            df_filtered['English Translation'] = df_filtered['German Keyword'].map(translations).fillna("-")

        # --- OUTPUT ---
        st.markdown("---")
        st.markdown("### 1. Strategic Seeds")
        cols = st.columns(len(synonyms))
        for i, syn in enumerate(synonyms):
            cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(f"2. The Master Matrix")
        
        # Reorder columns
        df_display = df_filtered[['German Keyword', 'English Translation', 'Gemma Score', 'Source Seed']]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Gemma Score": st.column_config.ProgressColumn(
                    "Relevance", format="%.2f", min_value=0, max_value=1.0
                )
            }
        )
        
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Strategy (CSV)", csv, "german_strategy.csv", "text/csv")
            
    else:
        st.warning("No keywords found.")

elif run_btn:
    st.error("Please provide both API Keys.")
