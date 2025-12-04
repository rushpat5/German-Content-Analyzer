import streamlit as st
import pandas as pd
import requests
import json
import time
import google.generativeai as genai
from pytrends.request import TrendReq
import plotly.express as px
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German SEO Strategist", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    h1, h2, h3 { color: #111; font-weight: 600; letter-spacing: -0.5px; }
    
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    .cluster-box { border: 1px solid #e1e4e8; background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .tech-note { font-size: 0.85rem; color: #57606a; background-color: #f6f8fa; border-left: 3px solid #0969da; padding: 12px; border-radius: 0 4px 4px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VECTOR ENGINE (EmbeddingGemma)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except Exception as e:
        st.error(f"Gemma Error: {e}")
        return None

def filter_and_cluster_gemma(df_keywords, seeds, threshold, hf_token):
    model = load_gemma_model(hf_token)
    if not model: return df_keywords
    
    from sentence_transformers import util
    
    # --- STEP 1: FILTERING ---
    seed_vecs = model.encode(seeds, prompt_name="STS", normalize_embeddings=True)
    candidates = df_keywords['German Keyword'].tolist()
    candidate_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    
    # Compare candidates to seeds
    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    max_scores, _ = torch.max(scores_matrix, dim=1)
    
    df_keywords['Relevance'] = max_scores.numpy()
    df_filtered = df_keywords[df_keywords['Relevance'] >= threshold].copy()
    
    if len(df_filtered) < 2:
        return df_filtered 

    # --- STEP 2: CLUSTERING ---
    cluster_vecs = model.encode(df_filtered['German Keyword'].tolist(), prompt_name="Clustering", normalize_embeddings=True)
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.85, 
        metric='euclidean', 
        linkage='ward'
    )
    cluster_ids = clustering.fit_predict(cluster_vecs)
    df_filtered['Cluster ID'] = cluster_ids
    
    # Name Clusters
    cluster_names = {}
    for cid in np.unique(cluster_ids):
        subset = df_filtered[df_filtered['Cluster ID'] == cid]
        head_term = sorted(subset['German Keyword'].tolist(), key=len)[0]
        cluster_names[cid] = head_term.title()
        
    df_filtered['Cluster Name'] = df_filtered['Cluster ID'].map(cluster_names)
    return df_filtered.sort_values('Cluster ID')

# -----------------------------------------------------------------------------
# 3. GENERATIVE ENGINE (Dynamic & Robust)
# -----------------------------------------------------------------------------

def get_working_model_name(api_key):
    """Automatically finds a model your API key has access to."""
    genai.configure(api_key=api_key)
    try:
        models = list(genai.list_models())
        # Filter for generating content
        valid_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models: return "models/gemini-1.5-flash"
        
        # Priority 1: Flash 1.5
        for m in valid_models:
            if 'flash' in m and '1.5' in m: return m
        # Priority 2: Pro 1.5
        for m in valid_models:
            if 'pro' in m and '1.5' in m: return m
            
        return valid_models[0] # Fallback to whatever is available
    except:
        return "models/gemini-1.5-flash"

def run_gemini(api_key, prompt):
    try:
        model_name = get_working_model_name(api_key)
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Clean Markdown wrappers
        if "```" in text:
            text = re.sub(r"```json|```", "", text).strip()
            
        return json.loads(text)
    except Exception as e:
        # Return the actual error message for debugging
        return {"error": str(e)}

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO. English Keyword: "{keyword}"
    Task: Identify the top 3 distinct German terms (Colloquial, Medical, Synonym).
    Return JSON: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Reasoning" }}
    """
    return run_gemini(api_key, prompt)

def batch_translate(api_key, keywords):
    if not keywords: return {}
    # Chunking
    chunks = [keywords[i:i+40] for i in range(0, len(keywords), 40)]
    full_map = {}
    
    progress_text = st.empty()
    for i, chunk in enumerate(chunks):
        progress_text.caption(f"Translating batch {i+1}/{len(chunks)}...")
        prompt = f"""Translate German to English (Literal): {json.dumps(chunk)}. 
        Return JSON: {{ "German": "English" }}"""
        res = run_gemini(api_key, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(0.5) # Safety sleep for rate limits
        
    progress_text.empty()
    return full_map

def generate_brief(api_key, cluster_name, keywords):
    prompt = f"""
    Act as a Content Strategist. Topic: "{cluster_name}". Keywords: {", ".join(keywords)}
    Create a German Content Brief.
    Return JSON:
    {{
        "h1_german": "Optimized H1",
        "h1_english": "Translation",
        "outline": [ {{ "h2": "German H2", "intent": "Notes" }} ]
    }}
    """
    return run_gemini(api_key, prompt)

# -----------------------------------------------------------------------------
# 4. MINING
# -----------------------------------------------------------------------------
def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine(synonyms):
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kaufen"]
    all_data = []
    
    prog = st.progress(0, "Mining Google Germany...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            prog.progress(min(step/total, 1.0), f"Mining: {seed}{mod}...")
            results = fetch_suggestions(f"{seed}{mod}")
            
            intent = "Informational"
            if "kaufen" in mod: intent = "Transactional"
            elif "gegen" in mod: intent = "Solution"
            
            for r in results:
                all_data.append({"German Keyword": r, "Seed": seed, "Intent": intent})
            time.sleep(0.05)
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty: return df.drop_duplicates(subset=['German Keyword'])
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
    
    st.markdown("""
    <div class="tech-note">
    <b>Dynamic AI:</b> Engine automatically finds valid models for your API Key region.
    </div>
    """, unsafe_allow_html=True)

st.title("German SEO Strategist 🇩🇪")
st.markdown("### Strategy Generator (Gemma + Clustering)")

keyword = st.text_input("Enter English Topic", placeholder="e.g. newborn babies")
run_btn = st.button("Generate Strategy", type="primary")

if run_btn and keyword and api_key and hf_token:
    
    # 0. Load
    with st.spinner("Loading Gemma..."):
        try: _ = load_gemma_model(hf_token)
        except: st.stop()

    # 1. Strategy
    with st.spinner("Analyzing Linguistics..."):
        strategy = get_cultural_translation(api_key, keyword)
        
        # ERROR HANDLING DISPLAY
        if not strategy or "error" in strategy:
            err_msg = strategy.get('error') if strategy else "Unknown Error"
            st.error(f"AI Connection Failed: {err_msg}")
            st.info("Tip: Check if your API Key has access to 'Gemini Flash' or 'Pro'.")
            st.stop()
    
    synonyms = strategy.get('synonyms', [])
    st.info(f"**Cultural Context:** {strategy.get('explanation')}")
    
    # 2. Mine
    df = deep_mine(synonyms)
    
    if not df.empty:
        # 3. FILTER & CLUSTER
        with st.spinner("Gemma Filtering & Clustering..."):
            df_clustered = filter_and_cluster_gemma(df, synonyms, threshold, hf_token)
            
            dropped = len(df) - len(df_clustered)
            st.success(f"Filtered {dropped} irrelevant keywords. Clustered {len(df_clustered)} relevant ones.")

        # 4. Translate
        with st.spinner("Translating..."):
            germ_list = df_clustered['German Keyword'].tolist()
            trans_map = batch_translate(api_key, germ_list)
            df_clustered['English'] = df_clustered['German Keyword'].map(trans_map).fillna("-")

        # --- OUTPUT ---
        tab1, tab2 = st.tabs(["🧠 Content Clusters", "📊 Master Data"])
        
        with tab1:
            st.markdown("### Topic Authority Map")
            clusters = df_clustered['Cluster Name'].unique()
            
            for c_name in clusters:
                c_data = df_clustered[df_clustered['Cluster Name'] == c_name]
                kws = c_data['German Keyword'].tolist()
                
                with st.expander(f"📁 {c_name} ({len(c_data)} keywords)"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.dataframe(c_data[['German Keyword', 'English', 'Relevance']], use_container_width=True, hide_index=True)
                    with c2:
                        if st.button(f"✨ Create Brief", key=c_name):
                            with st.spinner("Drafting..."):
                                brief = generate_brief(api_key, c_name, kws[:8])
                                if brief and "error" not in brief:
                                    st.success("Brief Ready!")
                                    st.markdown(f"**H1:** {brief.get('h1_german')}")
                                    st.caption(f"({brief.get('h1_english')})")
                                    for s in brief.get('outline', []):
                                        st.markdown(f"- {s.get('h2')}")
                                else:
                                    st.error(f"Brief Failed: {brief.get('error')}")
        
        with tab2:
            st.dataframe(df_clustered, use_container_width=True)
            csv = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "german_gemma_clusters.csv", "text/csv")
            
    else:
        st.warning("No keywords found.")

elif run_btn:
    st.error("Keys required.")
