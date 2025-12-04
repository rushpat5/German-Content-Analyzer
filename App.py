import streamlit as st
import pandas as pd
import requests
import json
import time
import google.generativeai as genai
from pytrends.request import TrendReq
import plotly.express as px
import re
from sklearn.cluster import AgglomerativeClustering
import numpy as np

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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VECTOR ENGINE (Cloud-Based / Lightweight)
# -----------------------------------------------------------------------------

def get_embeddings_google(text_list, api_key):
    """
    Uses Google's 'text-embedding-004' model via API.
    Zero RAM usage, instant speed.
    """
    genai.configure(api_key=api_key)
    
    # Batching (Google accepts up to 100 docs per call usually, we stick to safe batches)
    # Note: For 'text-embedding-004', content is passed as 'content'
    embeddings = []
    
    # Simple loop to avoid complexity, for SEO lists (usually < 200 items) this is fast enough
    # Ideally, use batch_embed_contents if lists are huge
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text_list,
            task_type="clustering",
        )
        # The result['embedding'] is a list of vectors
        return result['embedding']
    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return []

def filter_by_semantic_relevance(df_keywords, seed_english, threshold, api_key):
    # 1. Embed Seed
    seed_vec = get_embeddings_google([seed_english], api_key)[0]
    
    # 2. Embed Candidates
    candidates = df_keywords['German Keyword'].tolist()
    # We might need to batch this if > 100 keywords, but for this tool it's okay
    candidate_vecs = get_embeddings_google(candidates, api_key)
    
    if not candidate_vecs: return df_keywords # Fallback if API fails
    
    # 3. Cosine Similarity Manually (No heavy libraries)
    scores = []
    seed_norm = np.linalg.norm(seed_vec)
    
    for vec in candidate_vecs:
        vec_norm = np.linalg.norm(vec)
        if seed_norm == 0 or vec_norm == 0:
            scores.append(0.0)
        else:
            dot_product = np.dot(seed_vec, vec)
            scores.append(dot_product / (seed_norm * vec_norm))
            
    df_keywords['Relevance Score'] = scores
    
    # Filter
    df_filtered = df_keywords[df_keywords['Relevance Score'] >= threshold].copy()
    return df_filtered.sort_values('Relevance Score', ascending=False)

def cluster_keywords(df_keywords, api_key):
    if len(df_keywords) < 3:
        df_keywords['Cluster'] = "Single Topic"
        return df_keywords

    # Get Vectors via API
    embeddings = get_embeddings_google(df_keywords['German Keyword'].tolist(), api_key)
    
    if not embeddings: return df_keywords
    
    # Cluster
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.8, # Adjusted for Google Embeddings scale
        metric='euclidean', 
        linkage='ward'
    )
    cluster_ids = clustering.fit_predict(embeddings)
    df_keywords['Cluster ID'] = cluster_ids
    
    # Name Clusters
    cluster_names = {}
    for cid in np.unique(cluster_ids):
        subset = df_keywords[df_keywords['Cluster ID'] == cid]
        # Head term is usually the shortest
        head_term = sorted(subset['German Keyword'].tolist(), key=len)[0]
        cluster_names[cid] = head_term.title()
        
    df_keywords['Cluster Name'] = df_keywords['Cluster ID'].map(cluster_names)
    return df_keywords.sort_values('Cluster ID')

# -----------------------------------------------------------------------------
# 3. GENERATIVE ENGINE (Robust)
# -----------------------------------------------------------------------------

def run_gemini(api_key, prompt):
    genai.configure(api_key=api_key)
    candidates = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"]
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            if "429" in str(e): time.sleep(1)
            continue
    return {"error": "All models failed."}

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    Task: Identify the top 3 distinct German terms used for this concept.
    1. Colloquial (Most common).
    2. Formal/Medical.
    3. Synonym.
    Return JSON: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Reasoning" }}
    """
    return run_gemini(api_key, prompt)

def batch_validate_translate(api_key, keywords, topic):
    if not keywords: return {}
    full_map = {}
    chunks = [keywords[i:i + 30] for i in range(0, len(keywords), 30)]
    
    prog_text = st.empty()
    for i, chunk in enumerate(chunks):
        prog_text.caption(f"AI Validating Batch {i+1}/{len(chunks)}...")
        prompt = f"""
        Context Topic: "{topic}"
        Keywords: {json.dumps(chunk)}
        Task: 
        1. Translate to English.
        2. "keep": false if it is a Brand Name, Store, or completely Off-Topic.
        Return JSON: {{ "german_word": {{ "en": "english", "keep": true }} }}
        """
        res = run_gemini(api_key, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(0.2)
    prog_text.empty()
    return full_map

def generate_content_brief(api_key, cluster_name, keywords):
    prompt = f"""
    Act as a Content Strategist.
    Target Topic: "{cluster_name}"
    Keywords: {", ".join(keywords)}
    Create a brief for a German article.
    Return JSON:
    {{
        "h1_german": "Optimized H1",
        "h1_english": "Translation",
        "outline": [ {{ "h2": "German H2", "intent": "Content notes" }} ]
    }}
    """
    return run_gemini(api_key, prompt)

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
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " anleitung", " kaufen"]
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
                all_data.append({
                    "German Keyword": r, 
                    "Seed": seed,
                    "Intent": intent,
                    "Length": len(r)
                })
            time.sleep(0.05)
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        return df.drop_duplicates(subset=['German Keyword'])
    return df

def fetch_smart_trends(df_keywords):
    candidates = df_keywords.sort_values('Length').head(10)['German Keyword'].tolist()
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
# 5. UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Strategist Config")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("""<a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;color:#0969da;">🔑 Get Free Key</a>""", unsafe_allow_html=True)
    
    st.markdown("---")
    relevance_threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.55, 0.05)
    st.markdown(f"""
    <div class="tech-note">
    <b>Cloud Vector Engine:</b>
    Using Google's <code>text-embedding-004</code> for instant, lightweight clustering.
    </div>
    """, unsafe_allow_html=True)

st.title("German SEO Strategist 🇩🇪")
st.markdown("### Keyword Clustering & Content Architecture")

keyword = st.text_input("Enter English Topic", placeholder="e.g. newborn babies")
run_btn = st.button("Generate Strategy", type="primary")

if run_btn and keyword and api_key:
    
    # 1. Strategy
    with st.spinner("Analyzing German Linguistics..."):
        strategy = get_cultural_translation(api_key, keyword)
    
    if not strategy or "error" in strategy:
        st.error(f"AI Error: {strategy.get('error') if strategy else 'Unknown'}")
        st.stop()

    synonyms = strategy.get('synonyms', [])
    st.info(f"**Cultural Context:** {strategy.get('explanation')}")
    
    # 2. Mine
    df = deep_mine(synonyms)
    
    if not df.empty:
        # 3. Filter & Translate
        with st.spinner("Validating & Clustering Keywords (Cloud Mode)..."):
            
            # --- CLOUD SEMANTIC FILTERING ---
            df_filtered = filter_by_semantic_relevance(df, keyword, relevance_threshold, api_key)
            
            dropped_count = len(df) - len(df_filtered)
            if df_filtered.empty:
                st.error(f"All keywords filtered. Try lowering the threshold.")
                st.stop()
            
            st.success(f"Kept {len(df_filtered)} relevant keywords (Discarded {dropped_count}).")

            # Translate kept keywords
            raw_list = df_filtered['German Keyword'].tolist()
            valid_map = batch_validate_translate(api_key, raw_list, keyword)
            
            df_filtered['English'] = df_filtered['German Keyword'].apply(lambda x: valid_map.get(x, {}).get('en', '-'))
            df_filtered['Keep'] = df_filtered['German Keyword'].apply(lambda x: valid_map.get(x, {}).get('keep', True))
            
            df_clean = df_filtered[df_filtered['Keep'] == True].copy()
            
            if df_clean.empty:
                st.warning("AI filtered out all keywords (Brand/Irrelevant).")
                df_clean = df_filtered
            
            # 4. CLUSTERING (Via Cloud Embeddings)
            df_clustered = cluster_keywords(df_clean, api_key)
            
            # 5. Trends
            trends = fetch_smart_trends(df_clustered)
            df_clustered['Trend'] = df_clustered['German Keyword'].map(trends).fillna("-")

        # --- OUTPUT TABS ---
        tab_clusters, tab_data = st.tabs(["🧠 Topic Clusters", "📊 Raw Data"])
        
        with tab_clusters:
            st.markdown("### Content Architecture")
            clusters = df_clustered['Cluster Name'].unique()
            
            for c_name in clusters:
                c_data = df_clustered[df_clustered['Cluster Name'] == c_name]
                keywords_list = c_data['German Keyword'].tolist()
                
                with st.expander(f"📁 Cluster: {c_name} ({len(c_data)} keywords)"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.dataframe(c_data[['German Keyword', 'English', 'Relevance Score']], use_container_width=True, hide_index=True)
                    with c2:
                        if st.button(f"✨ Create Brief", key=c_name):
                            with st.spinner("Drafting..."):
                                brief = generate_content_brief(api_key, c_name, keywords_list[:8])
                                if brief and "error" not in brief:
                                    st.success("Brief Ready!")
                                    st.markdown(f"**H1 (DE):** {brief.get('h1_german')}")
                                    st.markdown(f"**H1 (EN):** {brief.get('h1_english')}")
                                    for sec in brief.get('outline', []):
                                        st.markdown(f"- **{sec.get('h2')}**")
                                else:
                                    st.error("Brief generation failed.")
        
        with tab_data:
            st.dataframe(df_clustered, use_container_width=True)
            csv = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button("Download Strategy CSV", csv, "german_clusters.csv", "text/csv")
            
    else:
        st.warning("No keywords found.")

elif run_btn:
    st.error("API Key required.")
