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
# 3. VECTOR ENGINE (Lightweight - Scoring Only)
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
    
    # 1. Encode Seeds (The English/German topic context)
    try:
        seed_vecs = model.encode(seeds, prompt_name="STS", normalize_embeddings=True)
        candidates = df_keywords['German Keyword'].tolist()
        candidate_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    except TypeError:
        # Fallback for models that don't support prompts
        seed_vecs = model.encode(seeds, normalize_embeddings=True)
        candidates = df_keywords['German Keyword'].tolist()
        candidate_vecs = model.encode(candidates, normalize_embeddings=True)
    
    # 2. Calculate Similarity Scores
    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    max_scores, _ = torch.max(scores_matrix, dim=1)
    
    df_keywords['Relevance'] = max_scores.numpy()
    
    # 3. Filter only relevant terms
    df_relevant = df_keywords[df_keywords['Relevance'] >= threshold].copy()
    
    return df_relevant.sort_values('Relevance', ascending=False)

# -----------------------------------------------------------------------------
# 4. GENERATIVE ENGINE (Optimized)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Connecting to Google AI...", ttl=3600)
def get_valid_gemini_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        if not valid: return None
        # Prefer Flash -> Pro -> Standard
        for m in valid:
            if 'flash' in m and '1.5' in m: return m
        for m in valid:
            if 'pro' in m and '1.5' in m: return m
        return valid[0]
    except: return None

def run_gemini(api_key, prompt, retries=0):
    if retries > 3: return {"error": "API Limit Hit"}

    model_name = get_valid_gemini_model(api_key) or "models/gemini-1.5-flash"
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", response.text).strip()
        return json.loads(text)
    except Exception as e:
        if "429" in str(e): 
            time.sleep(2 * (retries + 1))
            return run_gemini(api_key, prompt, retries+1)
        if "404" in str(e) and retries == 0:
            get_valid_gemini_model.clear()
            return run_gemini(api_key, prompt, retries+1)
        return {"error": str(e)}

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
    # Increased chunk size for speed, decreased calls
    chunks = [keywords[i:i+40] for i in range(0, len(keywords), 40)]
    full_map = {}
    for chunk in chunks:
        prompt = f"""Translate German to English (Literal): {json.dumps(chunk)}. 
        Return JSON: {{ "German": "English" }}"""
        res = run_gemini(api_key, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(0.5)
    return full_map

# -----------------------------------------------------------------------------
# 5. MINING LOGIC
# -----------------------------------------------------------------------------
def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=1.5) # Fast timeout
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine(synonyms):
    # Reduced modifiers for speed
    modifiers = ["", " für", " bei", " gegen", " hausmittel", " kaufen", " test"]
    all_data = []
    
    prog = st.progress(0, "Mining Google Autocomplete...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            if total > 0: prog.progress(min(step/total, 1.0), f"Scanning: {seed}{mod}")
            
            results = fetch_suggestions(f"{seed}{mod}")
            
            # Simple Intent Logic
            intent = "Informational"
            if "kaufen" in mod or "preis" in mod: intent = "Transactional"
            elif "gegen" in mod or "mittel" in mod: intent = "Commercial/Solution"
            elif "test" in mod or "vergleich" in mod: intent = "Commercial Investigation"
            
            for r in results:
                all_data.append({"German Keyword": r, "Seed": seed, "Intent": intent})
            time.sleep(0.05) # Polite scraping delay
            
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
    st.caption("Lower = More results. Higher = Stricter relevance.")

st.title("German SEO Planner 🇩🇪")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic", placeholder="e.g. coffee machines")
run_btn = st.button("Generate Keywords", type="primary")

if run_btn and keyword and api_key and hf_token:
    st.session_state.data_processed = False
    
    # 1. Load Model
    with st.spinner("Loading AI Models..."):
        try: _ = load_gemma_model(hf_token)
        except: st.stop()

    # 2. Strategy
    with st.spinner("Analyzing Cultural Context..."):
        strategy = get_cultural_translation(api_key, keyword)
        if not strategy or "error" in strategy:
            st.error("Gemini API Error. Check Key/Quota.")
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
            # 5. Translate (Only the good ones)
            with st.spinner("Translating Top Results..."):
                keywords_to_translate = df_filtered['German Keyword'].tolist()
                trans_map = batch_translate(api_key, keywords_to_translate)
                df_filtered['English'] = df_filtered['German Keyword'].map(trans_map).fillna("-")
            
            st.session_state.df_results = df_filtered
            st.session_state.data_processed = True
        else:
            st.warning("Keywords found, but none met the relevance threshold. Try lowering it.")
    else:
        st.warning("No keywords found via Autocomplete.")

# --- DISPLAY RESULTS ---
if st.session_state.data_processed:
    
    # Context
    st.success(f"**Cultural Insight:** {st.session_state.strategy_text}")
    cols = st.columns(len(st.session_state.synonyms))
    for i, syn in enumerate(st.session_state.synonyms):
        if i < len(cols):
            cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Verified Keyword List")
    
    df_show = st.session_state.df_results
    
    # Download Button
    csv = df_show.to_csv(index=False).encode('utf-8')
    c1, c2 = st.columns([4,1])
    with c2:
        st.download_button("📥 Download CSV", csv, "german_keywords.csv", "text/csv", use_container_width=True)
    
    # Data Table
    st.dataframe(
        df_show[['German Keyword', 'English', 'Intent', 'Relevance']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relevance": st.column_config.ProgressColumn("Vector Match", format="%.2f", min_value=0, max_value=1),
            "Intent": st.column_config.TextColumn("Search Intent", width="medium")
        }
    )

elif not st.session_state.data_processed and run_btn:
    st.error("Please provide both API Keys.")
