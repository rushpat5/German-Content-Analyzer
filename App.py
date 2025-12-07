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
# 1. VISUAL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German SEO Planner (Final)", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-val { font-size: 1.2rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE
# -----------------------------------------------------------------------------
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_results = None
    st.session_state.synonyms = []
    st.session_state.strategy_text = ""

# -----------------------------------------------------------------------------
# 3. VECTOR ENGINE (Lightweight)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except Exception as e:
        return SentenceTransformer("all-MiniLM-L6-v2")

def process_keywords_gemma(df_keywords, seeds, threshold, hf_token):
    model = load_gemma_model(hf_token)
    if not model: return None
    
    try:
        seed_vecs = model.encode(seeds, prompt_name="STS", normalize_embeddings=True)
        candidates = df_keywords['German Keyword'].tolist()
        candidate_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    except TypeError:
        seed_vecs = model.encode(seeds, normalize_embeddings=True)
        candidates = df_keywords['German Keyword'].tolist()
        candidate_vecs = model.encode(candidates, normalize_embeddings=True)
    
    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    max_scores, _ = torch.max(scores_matrix, dim=1)
    
    df_keywords['Relevance'] = max_scores.numpy()
    return df_keywords[df_keywords['Relevance'] >= threshold].sort_values('Relevance', ascending=False)

# -----------------------------------------------------------------------------
# 4. GENERATIVE ENGINE (CACHED MODEL DISCOVERY)
# -----------------------------------------------------------------------------

# This function runs ONCE. It asks Google what models you actually have.
# It solves the 404 error by getting the EXACT name your key supports.
@st.cache_resource(show_spinner="Connecting to Google...", ttl=3600)
def find_best_model(api_key):
    genai.configure(api_key=api_key)
    try:
        # 1. Ask Google: "What models do I have?"
        all_models = list(genai.list_models())
        
        # 2. Filter: Only models that generate text
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return "ERROR_NO_MODELS"
            
        # 3. Smart Selection (Priority: Flash -> Pro -> Standard)
        # We look for specific strings in the names returned by Google
        for m in valid_models:
            if 'flash' in m and '1.5' in m: return m
        for m in valid_models:
            if 'pro' in m and '1.5' in m: return m
        for m in valid_models:
            if 'gemini-pro' in m: return m
            
        # 4. Fallback: Just take the first one available
        return valid_models[0]
        
    except Exception as e:
        err_str = str(e)
        if "429" in err_str: return "ERROR_QUOTA"
        if "API key" in err_str: return "ERROR_KEY"
        return f"ERROR_UNKNOWN: {err_str}"

def run_gemini(api_key, model_name, prompt):
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", response.text).strip()
        return json.loads(text)
    except Exception as e:
        # Simple retry for transient network issues
        if "429" in str(e):
            time.sleep(5)
            try:
                response = model.generate_content(prompt)
                text = re.sub(r"```json|```", "", response.text).strip()
                return json.loads(text)
            except: pass
        return {"error": str(e)}

def get_cultural_translation(api_key, model_name, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term.
    2. The formal/medical term.
    3. A popular synonym.
    Return STRICT JSON: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Brief reasoning" }}
    """
    return run_gemini(api_key, model_name, prompt)

def batch_translate(api_key, model_name, keywords):
    if not keywords: return {}
    chunks = [keywords[i:i+50] for i in range(0, len(keywords), 50)]
    full_map = {}
    for chunk in chunks:
        prompt = f"""Translate German to English (Literal): {json.dumps(chunk)}. 
        Return JSON: {{ "German": "English" }}"""
        res = run_gemini(api_key, model_name, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(1)
    return full_map

# -----------------------------------------------------------------------------
# 5. MINING & MAIN APP
# -----------------------------------------------------------------------------
def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=1.0)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine(synonyms):
    modifiers = ["", " kaufen", " test", " vergleich", " kosten"]
    all_data = []
    prog = st.progress(0, "Mining Google...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            if total > 0: prog.progress(min(step/total, 1.0))
            
            results = fetch_suggestions(f"{seed}{mod}")
            intent = "Informational"
            if "kaufen" in mod or "kosten" in mod: intent = "Transactional"
            elif "test" in mod or "vergleich" in mod: intent = "Commercial"
            
            for r in results:
                all_data.append({"German Keyword": r, "Seed": seed, "Intent": intent})
            time.sleep(0.05)
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty: return df.drop_duplicates(subset=['German Keyword'])
    return df

# --- UI ---
with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    st.markdown("---")
    threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.45, 0.05)
    
    if st.button("🔄 Reset Connection"):
        st.cache_resource.clear()
        st.experimental_rerun()

st.title("German SEO Planner 🇩🇪")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic", placeholder="e.g. coffee machines")
run_btn = st.button("Generate Keywords", type="primary")

if run_btn and keyword and api_key and hf_token:
    st.session_state.data_processed = False
    
    # 1. ESTABLISH MODEL CONNECTION (Robust)
    with st.spinner("Connecting to Google AI (One-time check)..."):
        model_name = find_best_model(api_key)
        
        # ERROR HANDLING FOR CONNECTION
        if model_name == "ERROR_QUOTA":
            st.error("⚠️ API Quota Limit Hit (429).")
            st.warning("Please wait 60 seconds, then click 'Reset Connection' in the sidebar.")
            st.stop()
        elif model_name == "ERROR_KEY":
            st.error("❌ Invalid API Key.")
            st.stop()
        elif "ERROR" in model_name:
            st.error(f"❌ Connection Failed: {model_name}")
            st.stop()
        
        st.success(f"Connected to: {model_name}")

    # 2. LOAD VECTORS
    with st.spinner("Loading Vector Model..."):
        try: _ = load_gemma_model(hf_token)
        except: st.stop()

    # 3. STRATEGY
    with st.spinner("Analyzing Cultural Context..."):
        strategy = get_cultural_translation(api_key, model_name, keyword)
        
        if not strategy or "error" in strategy:
            st.error(f"Generation Error: {strategy.get('error')}")
            st.stop()
            
        st.session_state.synonyms = strategy.get('synonyms', [])
        st.session_state.strategy_text = strategy.get('explanation', '')

    # 4. MINE
    df = deep_mine(st.session_state.synonyms)
    
    if not df.empty:
        with st.spinner(f"Filtering {len(df)} keywords..."):
            df_filtered = process_keywords_gemma(df, st.session_state.synonyms, threshold, hf_token)
            
        if df_filtered is not None and not df_filtered.empty:
            with st.spinner("Translating..."):
                top = df_filtered.head(30)['German Keyword'].tolist()
                trans_map = batch_translate(api_key, model_name, top)
                df_filtered['English'] = df_filtered['German Keyword'].map(trans_map).fillna("-")
            
            st.session_state.df_results = df_filtered
            st.session_state.data_processed = True
        else: st.warning("No keywords met threshold.")
    else: st.warning("No keywords found.")

if st.session_state.data_processed:
    st.success(f"**Context:** {st.session_state.strategy_text}")
    cols = st.columns(len(st.session_state.synonyms))
    for i, syn in enumerate(st.session_state.synonyms):
        if i < len(cols): cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
    
    df_show = st.session_state.df_results
    csv = df_show.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "keywords.csv", "text/csv")
    
    st.dataframe(df_show[['German Keyword', 'English', 'Intent', 'Relevance']], use_container_width=True, hide_index=True, column_config={"Relevance": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=1)})
