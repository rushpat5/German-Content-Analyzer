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
st.set_page_config(page_title="German SEO Planner (Lite)", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    .metric-val { font-size: 1.2rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
        st.warning(f"Gemma Load Error: {e}. Using fast fallback model.")
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
    df_relevant = df_keywords[df_keywords['Relevance'] >= threshold].copy()
    
    return df_relevant.sort_values('Relevance', ascending=False)

# -----------------------------------------------------------------------------
# 4. GENERATIVE ENGINE (CACHED DISCOVERY)
# -----------------------------------------------------------------------------

# This function runs ONCE per hour. It finds the correct model name
# without wasting your API quota on every click.
@st.cache_data(ttl=3600, show_spinner="Connecting to Google AI...")
def find_working_model(api_key):
    genai.configure(api_key=api_key)
    try:
        # List all models available to your specific key
        all_models = list(genai.list_models())
        
        # Filter for models that can generate text
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models:
            return None
            
        # Select the best one: Flash -> Pro -> Standard
        for m in valid_models:
            if 'flash' in m and '1.5' in m: return m
        for m in valid_models:
            if 'pro' in m and '1.5' in m: return m
        for m in valid_models:
            if 'gemini' in m: return m
            
        return valid_models[0]
        
    except Exception:
        # If listing fails (e.g. strict quota), return None to trigger fallback
        return None

def run_gemini(api_key, prompt, retries=0):
    if retries > 3:
        return {"error": "⚠️ API limit hit. Please wait a moment."}

    # 1. Get the correct model name (Cached)
    model_name = find_working_model(api_key)
    
    # 2. Fallback if discovery failed (e.g. due to quota on the listing call)
    if not model_name:
        model_name = "models/gemini-1.5-flash"

    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", response.text).strip()
        return json.loads(text)

    except Exception as e:
        last_error = str(e)
        
        # Rate Limit (429): Wait and retry
        if "429" in last_error:
            time.sleep(5) 
            return run_gemini(api_key, prompt, retries=retries+1)
        
        # Model Not Found (404): Clear cache and retry once
        # This handles cases where the key permissions changed
        if "404" in last_error and retries == 0:
            find_working_model.clear() 
            return run_gemini(api_key, prompt, retries=retries+1)

        return {"error": f"AI Error ({model_name}): {last_error}"}

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term.
    2. The formal/medical term.
    3. A popular synonym.
    Return STRICT JSON: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Brief reasoning" }}
    """
    return run_gemini(api_key, prompt)

def batch_translate(api_key, keywords):
    if not keywords: return {}
    # Increased chunk size to save calls
    chunks = [keywords[i:i+50] for i in range(0, len(keywords), 50)]
    full_map = {}
    for chunk in chunks:
        prompt = f"""Translate German to English (Literal): {json.dumps(chunk)}. 
        Return JSON: {{ "German": "English" }}"""
        res = run_gemini(api_key, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(1)
    return full_map

# -----------------------------------------------------------------------------
# 5. MINING LOGIC
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
    
    prog = st.progress(0, "Mining Google Autocomplete...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            if total > 0: prog.progress(min(step/total, 1.0), f"Scanning: {seed}{mod}")
            
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

# -----------------------------------------------------------------------------
# 6. MAIN APP
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    st.markdown("---")
    threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.45, 0.05)
    
    if st.button("Reset Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

st.title("German SEO Planner 🇩🇪")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic", placeholder="e.g. coffee machines")
run_btn = st.button("Generate Keywords", type="primary")

if run_btn and keyword and api_key and hf_token:
    st.session_state.data_processed = False
    
    # 1. Load Model
    with st.spinner("Loading Vector Model..."):
        try: _ = load_gemma_model(hf_token)
        except: st.stop()

    # 2. Strategy
    with st.spinner("Analyzing Cultural Context..."):
        strategy = get_cultural_translation(api_key, keyword)
        
        if not strategy:
            st.error("No response from Gemini.")
            st.stop()
        if "error" in strategy:
            st.error(f"{strategy['error']}")
            st.warning("Tip: If you see a 404, check your API key permissions.")
            st.stop()
            
        st.session_state.synonyms = strategy.get('synonyms', [])
        st.session_state.strategy_text = strategy.get('explanation', '')

    # 3. Mine
    df = deep_mine(st.session_state.synonyms)
    
    if not df.empty:
        # 4. Filter
        with st.spinner(f"Vector Filtering {len(df)} Keywords..."):
            df_filtered = process_keywords_gemma(df, st.session_state.synonyms, threshold, hf_token)
            
        if df_filtered is not None and not df_filtered.empty:
            # 5. Translate (Limit to Top 30)
            with st.spinner("Translating Top Results..."):
                top_keywords = df_filtered.head(30)['German Keyword'].tolist()
                trans_map = batch_translate(api_key, top_keywords)
                df_filtered['English'] = df_filtered['German Keyword'].map(trans_map).fillna("-")
            
            st.session_state.df_results = df_filtered
            st.session_state.data_processed = True
        else:
            st.warning("No relevant keywords found. Lower the threshold.")
    else:
        st.warning("No keywords found via Autocomplete.")

# --- DISPLAY RESULTS ---
if st.session_state.data_processed:
    
    st.success(f"**Insight:** {st.session_state.strategy_text}")
    cols = st.columns(len(st.session_state.synonyms))
    for i, syn in enumerate(st.session_state.synonyms):
        if i < len(cols):
            cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Verified Keyword List")
    
    df_show = st.session_state.df_results
    
    csv = df_show.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "german_keywords.csv", "text/csv")
    
    st.dataframe(
        df_show[['German Keyword', 'English', 'Intent', 'Relevance']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relevance": st.column_config.ProgressColumn("Vector Match", format="%.2f", min_value=0, max_value=1),
        }
    )

elif not st.session_state.data_processed and run_btn:
    st.error("Please provide both API Keys.")
