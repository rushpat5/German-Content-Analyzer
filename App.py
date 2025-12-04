import streamlit as st
import pandas as pd
import requests
import json
import time
import google.generativeai as genai
from pytrends.request import TrendReq
import plotly.express as px
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION (Strict Dejan Style)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German Vector Strategist", layout="wide", page_icon="🇩🇪")

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
# 2. SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_clustered = None
    st.session_state.df_direct = None 
    st.session_state.synonyms = []
    st.session_state.strategy_text = ""
    st.session_state.briefs = {} 

# -----------------------------------------------------------------------------
# 3. VECTOR ENGINE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except Exception as e:
        st.error(f"Gemma Error: {e}")
        return None

def process_keywords_gemma(df_keywords, seeds, threshold, hf_token):
    """
    1. Filters keywords.
    2. Separates "Direct Variations" (>0.85 score) from "Broad Clusters".
    3. Clusters the broad list.
    """
    model = load_gemma_model(hf_token)
    if not model: return None, None
    
    # --- A. SCORING ---
    seed_vecs = model.encode(seeds, prompt_name="STS", normalize_embeddings=True)
    candidates = df_keywords['German Keyword'].tolist()
    candidate_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    
    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    max_scores, _ = torch.max(scores_matrix, dim=1)
    
    df_keywords['Relevance'] = max_scores.numpy()
    
    # Filter noise
    df_relevant = df_keywords[df_keywords['Relevance'] >= threshold].copy()
    
    # --- B. SPLITTING (Direct vs Broad) ---
    # Direct Variations: Extremely high similarity (>0.82) OR contains exact seed word
    df_direct = df_relevant[df_relevant['Relevance'] > 0.82].copy()
    df_clusters = df_relevant[df_relevant['Relevance'] <= 0.82].copy()
    
    # --- C. CLUSTERING (Only the broad ones) ---
    if len(df_clusters) > 2:
        cluster_vecs = model.encode(df_clusters['German Keyword'].tolist(), prompt_name="Clustering", normalize_embeddings=True)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.85, metric='euclidean', linkage='ward')
        cluster_ids = clustering.fit_predict(cluster_vecs)
        df_clusters['Cluster ID'] = cluster_ids
        
        # Name clusters
        cluster_names = {}
        for cid in np.unique(cluster_ids):
            subset = df_clusters[df_clusters['Cluster ID'] == cid]
            head_term = sorted(subset['German Keyword'].tolist(), key=len)[0]
            cluster_names[cid] = head_term.title()
        df_clusters['Cluster Name'] = df_clusters['Cluster ID'].map(cluster_names)
    else:
        df_clusters['Cluster Name'] = "General"
        # Handle empty case if everything went to Direct
        if 'Cluster ID' not in df_clusters.columns: df_clusters['Cluster ID'] = 0

    return df_direct.sort_values('Relevance', ascending=False), df_clusters.sort_values('Cluster ID')

# -----------------------------------------------------------------------------
# 4. GENERATIVE ENGINE
# -----------------------------------------------------------------------------
def run_gemini(api_key, prompt):
    genai.configure(api_key=api_key)
    # Dynamic Model Selection
    try:
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        # Priority: Flash -> Pro
        chosen = next((m for m in valid if 'flash' in m), next((m for m in valid if 'pro' in m), "models/gemini-1.5-flash"))
        
        model = genai.GenerativeModel(chosen)
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        if "429" in str(e): time.sleep(1); return run_gemini(api_key, prompt) # Simple retry
        return {"error": str(e)}

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a German Market Specialist. English Input: "{keyword}"
    Identify 3 distinct German search terms:
    1. High-Volume Colloquial Term.
    2. Formal/Technical Term.
    3. Lateral Synonym/Variation.
    Return JSON: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Linguistic context analysis" }}
    """
    return run_gemini(api_key, prompt)

def batch_translate(api_key, keywords):
    if not keywords: return {}
    chunks = [keywords[i:i+50] for i in range(0, len(keywords), 50)]
    full_map = {}
    for chunk in chunks:
        prompt = f"""Translate German to English (Literal): {json.dumps(chunk)}. 
        Return JSON: {{ "German": "English" }}"""
        res = run_gemini(api_key, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(0.2)
    return full_map

def generate_brief(api_key, cluster_name, keywords):
    prompt = f"""
    Act as a Technical Content Strategist. Topic: "{cluster_name}". Keywords: {", ".join(keywords)}
    Create a structured German Content Brief.
    Return JSON:
    {{
        "h1_german": "Optimized German H1",
        "h1_english": "Literal English Translation",
        "user_intent": "Search Intent Analysis",
        "outline": [ {{ "h2": "German Subheading", "intent": "Technical notes for the writer" }} ]
    }}
    """
    return run_gemini(api_key, prompt)

# -----------------------------------------------------------------------------
# 5. MINING & TRENDS
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
    prog = st.progress(0, "Mining Data...")
    total = len(synonyms) * len(modifiers)
    step = 0
    for seed in synonyms:
        for mod in modifiers:
            step += 1; prog.progress(min(step/total, 1.0), f"Mining: {seed}{mod}...")
            results = fetch_suggestions(f"{seed}{mod}")
            for r in results:
                all_data.append({"German Keyword": r, "Seed": seed})
            time.sleep(0.05)
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty: return df.drop_duplicates(subset=['German Keyword'])
    return df

def fetch_smart_trends(df_keywords):
    candidates = df_keywords.sort_values(by="German Keyword", key=lambda x: x.str.len()).head(10)['German Keyword'].tolist()
    trend_map = {}
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        pytrends.build_payload(candidates[:5], cat=0, timeframe='today 3-m', geo='DE')
        data = pytrends.interest_over_time()
        if not data.empty:
            means = data.mean()
            for kw in candidates[:5]:
                if kw in means: trend_map[kw] = int(means[kw])
    except: pass
    return trend_map

# -----------------------------------------------------------------------------
# 6. UI & MAIN LOGIC
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Architecture")
    api_key = st.text_input("Gemini API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    
    st.markdown("""
    <a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;">🔑 Gemini Key</a> | 
    <a href="https://huggingface.co/settings/tokens" target="_blank" style="font-size:0.8rem;">🤗 HF Token</a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    threshold = st.slider("Semantic Gate (Threshold)", 0.0, 1.0, 0.50, 0.05)
    st.markdown(f"""
    <div class="tech-note">
    <b>Vector Similarity Threshold:</b>
    <br>Controls the strictness of the vector filter.
    <br>• <b>High (>0.65):</b> Only accepts keywords that are semantically identical to the input.
    <br>• <b>Moderate (0.50):</b> Accepts related concepts and lateral topics.
    </div>
    """, unsafe_allow_html=True)

st.title("German Vector Strategist 🇩🇪")
st.markdown("### Cross-Border Market Intelligence System")

with st.expander("Technical Methodology (Neural Retrieval)", expanded=False):
    st.markdown("""
    **1. Linguistic Mapping (Symbolic AI):**
    Instead of literal translation (which fails on intent), we use Large Language Models (LLMs) to map English concepts to their native German equivalents (e.g., "Mobile Phone" $\\to$ "Handy").
    
    **2. Recursive Mining (Data Retrieval):**
    We execute a recursive scrape of Google Autocomplete using German-specific modifiers (*für, gegen, bei*) to uncover high-intent long-tail queries that standard tools often miss.
    
    **3. Vector Filtering (Connectionist AI):**
    To ensure relevance, we embed all mined keywords into a 768-dimensional vector space using `google/embeddinggemma-300m`. We perform Cosine Similarity analysis against the seed terms to mathematically validate semantic alignment.
    """)

col_in, col_btn = st.columns([3, 1])
with col_in:
    keyword = st.text_input("Input Concept (English)", placeholder="e.g. project management software", label_visibility="collapsed")
with col_btn:
    run_btn = st.button("Initialize Analysis", type="primary", use_container_width=True)

# --- EXECUTION LOGIC ---
if run_btn and keyword and api_key and hf_token:
    # Reset State
    st.session_state.data_processed = False
    st.session_state.briefs = {}

    with st.spinner("Initializing Neural Engines (Gemma-300m)..."):
        try: _ = load_gemma_model(hf_token)
        except: st.error("Model Load Failed. Check HF Token."); st.stop()

    # 1. Strategy
    with st.spinner("Processing Linguistic Context..."):
        strategy = get_cultural_translation(api_key, keyword)
        if not strategy or "error" in strategy: st.error("AI Error."); st.stop()
        st.session_state.synonyms = strategy.get('synonyms', [])
        st.session_state.strategy_text = strategy.get('explanation', '')

    # 2. Mining
    df = deep_mine(st.session_state.synonyms)
    
    if not df.empty:
        # 3. Filter & Cluster
        with st.spinner("Calculating Vector Distances & Clustering..."):
            df_direct, df_clustered = process_keywords_gemma(df, st.session_state.synonyms, threshold, hf_token)
            
        # 4. Translate
        with st.spinner("Translating Data Matrix..."):
            all_kws = []
            if not df_direct.empty: all_kws.extend(df_direct['German Keyword'].tolist())
            if not df_clustered.empty: all_kws.extend(df_clustered['German Keyword'].tolist())
            
            trans_map = batch_translate(api_key, list(set(all_kws)))
            
            if not df_direct.empty:
                df_direct['English'] = df_direct['German Keyword'].map(trans_map).fillna("-")
            
            if not df_clustered.empty:
                df_clustered['English'] = df_clustered['German Keyword'].map(trans_map).fillna("-")
                trends = fetch_smart_trends(df_clustered)
                df_clustered['Trend'] = df_clustered['German Keyword'].map(trends).fillna("-")

        st.session_state.df_direct = df_direct
        st.session_state.df_clustered = df_clustered
        st.session_state.data_processed = True
    else:
        st.warning("No data returned from mining layer.")

# --- RENDER RESULTS FROM STATE ---
if st.session_state.data_processed:
    
    st.markdown("""
    <div class="tech-note" style="border-left-color: #1a7f37; background: #e6ffed;">
    <b>🧠 Linguistic Analysis:</b><br>
    """ + st.session_state.strategy_text + "</div>", unsafe_allow_html=True)

    cols = st.columns(len(st.session_state.synonyms))
    for i, syn in enumerate(st.session_state.synonyms):
        cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div><div class="metric-lbl">Primary Seed {i+1}</div></div>""", unsafe_allow_html=True)
    
    st.write("---")
    
    # SECTION 1: DIRECT
    st.subheader("1. High-Fidelity Variations")
    st.markdown("Keywords with **> 0.85 Vector Similarity**. These represent the same intent as the seed terms and should be treated as direct synonyms.")
    
    if st.session_state.df_direct is not None and not st.session_state.df_direct.empty:
        st.dataframe(
            st.session_state.df_direct[['German Keyword', 'English', 'Relevance']],
            use_container_width=True,
            hide_index=True,
            column_config={"Relevance": st.column_config.ProgressColumn("Vector Match", format="%.2f", min_value=0, max_value=1)}
        )
    else:
        st.info("No exact vector matches found. Intent is highly specific.")

    # SECTION 2: CLUSTERS
    st.markdown("---")
    st.subheader("2. Semantic Content Clusters")
    st.markdown("These groups represent distinct **sub-topics**. Each cluster indicates a separate user intent that justifies a unique URL or content section.")
    
    if st.session_state.df_clustered is not None and not st.session_state.df_clustered.empty:
        clusters = st.session_state.df_clustered['Cluster Name'].unique()
        
        for c_name in clusters:
            c_data = st.session_state.df_clustered[st.session_state.df_clustered['Cluster Name'] == c_name]
            keywords_list = c_data['German Keyword'].tolist()
            
            # Get English Translation of Cluster Name from the first keyword row
            english_title = c_data.iloc[0]['English'] if not c_data.empty else "Unknown"
            
            with st.expander(f"📁 {c_name} ({english_title})"):
                c1, c2 = st.columns([2, 1])
                with c1:
                    # ADDED 'Relevance' HERE
                    st.dataframe(
                        c_data[['German Keyword', 'English', 'Trend', 'Relevance']], 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Relevance": st.column_config.ProgressColumn("Relevance", format="%.2f", min_value=0, max_value=1)
                        }
                    )
                with c2:
                    # Brief Generation Logic
                    if st.button(f"✨ Draft Content Brief", key=f"btn_{c_name}"):
                        with st.spinner("Generating Technical Brief..."):
                            brief = generate_brief(api_key, c_name, keywords_list[:8])
                            st.session_state.briefs[c_name] = brief
                    
                    # Show stored brief if exists
                    if c_name in st.session_state.briefs:
                        b = st.session_state.briefs[c_name]
                        if "error" not in b:
                            st.success("Brief Generated")
                            st.markdown(f"**H1 (DE):** {b.get('h1_german')}")
                            st.caption(f"**H1 (EN):** {b.get('h1_english')}")
                            st.markdown(f"**Intent:** {b.get('user_intent')}")
                            st.markdown("**Outline:**")
                            for s in b.get('outline', []):
                                st.markdown(f"- {s.get('h2')}")
                        else:
                            st.error("Generation failed.")

        # Export
        st.markdown("---")
        csv = st.session_state.df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Data Matrix (CSV)", csv, "german_clusters.csv", "text/csv")

elif not st.session_state.data_processed and run_btn:
    st.error("Authentication Error: Please provide valid API Keys in the sidebar.")
