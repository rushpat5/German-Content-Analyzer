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

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German SEO Strategist", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    h1, h2, h3 { color: #111; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Cards */
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Clusters */
    .cluster-box { border: 1px solid #d0d7de; border-radius: 6px; padding: 15px; margin-bottom: 15px; background: #fcfcfc; }
    .cluster-header { font-weight: 700; font-size: 1.1rem; color: #0969da; border-bottom: 1px solid #eaecef; padding-bottom: 8px; margin-bottom: 8px; }
    
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VECTOR ENGINE (For Clustering)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    # Multilingual model crucial for German semantics
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def cluster_keywords(df_keywords):
    """
    Groups keywords by semantic meaning using Vector Embeddings.
    """
    if len(df_keywords) < 3:
        df_keywords['Cluster'] = "Single Topic"
        return df_keywords

    model = load_embedding_model()
    
    # 1. Vectorize German Keywords
    embeddings = model.encode(df_keywords['German Keyword'].tolist())
    
    # 2. Cluster (Agglomerative is better for small datasets than K-Means)
    # Distance threshold determines how "tight" the groups are
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=1.5, # Tweak this for tighter/looser groups
        metric='euclidean', 
        linkage='ward'
    )
    cluster_ids = clustering.fit_predict(embeddings)
    
    df_keywords['Cluster ID'] = cluster_ids
    
    # 3. Name the Clusters (Find the shortest keyword in the group as the "Head")
    cluster_names = {}
    for cid in np.unique(cluster_ids):
        subset = df_keywords[df_keywords['Cluster ID'] == cid]
        # Heuristic: Shortest keyword usually represents the broad topic
        head_term = sorted(subset['German Keyword'].tolist(), key=len)[0]
        cluster_names[cid] = head_term.title()
        
    df_keywords['Cluster Name'] = df_keywords['Cluster ID'].map(cluster_names)
    return df_keywords.sort_values('Cluster ID')

# -----------------------------------------------------------------------------
# 3. GENERATIVE ENGINE (Gemini)
# -----------------------------------------------------------------------------

def get_valid_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        for m in valid:
            if 'flash' in m and '1.5' in m: return m
        return "models/gemini-pro"
    except: return "models/gemini-1.5-flash"

def run_gemini(api_key, prompt):
    model_name = get_valid_model(api_key)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        # Return raw text if JSON parse fails, formatted as error dict
        if "429" in str(e): return {"error": "Quota Exceeded"}
        return {"error": str(e)}

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
    chunks = [keywords[i:i + 40] for i in range(0, len(keywords), 40)]
    full_map = {}
    
    for chunk in chunks:
        prompt = f"""
        Topic: "{topic}"
        Input Keywords: {json.dumps(chunk)}
        
        Task:
        1. Translate to English.
        2. Mark "keep": false if it is a Brand Name (e.g. DM, Rossmann), Store, or Off-Topic.
        
        Return JSON: {{ "german_word": {{ "en": "english", "keep": true }} }}
        """
        try:
            res = run_gemini(api_key, prompt)
            if "error" not in res: full_map.update(res)
            time.sleep(0.5)
        except: continue
    return full_map

def generate_content_brief(api_key, cluster_name, keywords):
    prompt = f"""
    Act as a Content Strategist for the German Market.
    Target Topic: "{cluster_name}"
    Keywords to Cover: {", ".join(keywords)}
    
    Create a Content Brief.
    Return JSON:
    {{
        "h1_german": "Optimized H1 in German",
        "h1_english": "English translation",
        "user_intent": "Informational/Commercial/Transactional",
        "outline": [
            {{ "h2": "German H2", "intent": "What to cover here" }}
        ]
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
    # Recursive Alphabet Soup
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " anleitung", " kaufen"]
    all_data = []
    
    # UI Progress
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
                    "Seed": seed,
                    "Length": len(r)
                })
            time.sleep(0.05)
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        return df.drop_duplicates(subset=['German Keyword'])
    return df

def fetch_smart_trends(df_keywords):
    # Top 10 Shortest (Head Terms)
    candidates = df_keywords.sort_values('Length').head(10)['German Keyword'].tolist()
    trend_map = {}
    
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        # Batch of 5
        for i in range(0, len(candidates), 5):
            batch = candidates[i:i+5]
            pytrends.build_payload(batch, cat=0, timeframe='today 3-m', geo='DE')
            data = pytrends.interest_over_time()
            if not data.empty:
                means = data.mean()
                for kw in batch:
                    if kw in means: trend_map[kw] = int(means[kw])
            time.sleep(1.0)
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
    st.markdown("""
    <div class="tech-note">
    <b>Pipeline:</b>
    <br>1. <b>Translation:</b> Finds cultural equivalents.
    <br>2. <b>Mining:</b> Scrapes Google Autocomplete.
    <br>3. <b>Clustering:</b> Uses Vector Embeddings to group keywords into "Content Clusters".
    <br>4. <b>Briefing:</b> Generates AI content outlines.
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
        st.error("AI Error. Check API Key.")
        st.stop()

    synonyms = strategy.get('synonyms', [])
    st.info(f"**Cultural Context:** {strategy.get('explanation')}")
    
    # 2. Mine
    df = deep_mine(synonyms)
    
    if not df.empty:
        # 3. Filter & Translate
        with st.spinner("Validating & Clustering Keywords..."):
            raw_list = df['German Keyword'].tolist()
            valid_map = batch_validate_translate(api_key, raw_list, keyword)
            
            df['English'] = df['German Keyword'].apply(lambda x: valid_map.get(x, {}).get('en', '-'))
            df['Keep'] = df['German Keyword'].apply(lambda x: valid_map.get(x, {}).get('keep', True))
            
            df_clean = df[df['Keep'] == True].copy()
            
            # 4. CLUSTERING (The Perfect Feature)
            df_clustered = cluster_keywords(df_clean)
            
            # 5. Trends
            trends = fetch_smart_trends(df_clustered)
            df_clustered['Trend'] = df_clustered['German Keyword'].map(trends).fillna("N/A")

        # --- OUTPUT TABS ---
        tab_clusters, tab_data = st.tabs(["🧠 Topic Clusters", "📊 Raw Data"])
        
        with tab_clusters:
            st.markdown("### Content Architecture")
            st.markdown("We grouped the keywords into semantic clusters. Each cluster represents **One Article** you should write.")
            
            clusters = df_clustered['Cluster Name'].unique()
            
            for c_name in clusters:
                c_data = df_clustered[df_clustered['Cluster Name'] == c_name]
                keywords_list = c_data['German Keyword'].tolist()
                
                with st.expander(f"📁 Cluster: {c_name} ({len(c_data)} keywords)"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.dataframe(c_data[['German Keyword', 'English', 'Trend']], use_container_width=True, hide_index=True)
                    
                    with c2:
                        if st.button(f"✨ Draft Brief for '{c_name}'", key=c_name):
                            with st.spinner("Generating Content Brief..."):
                                brief = generate_content_brief(api_key, c_name, keywords_list[:10])
                                if brief and "error" not in brief:
                                    st.success("Brief Generated!")
                                    st.markdown(f"**H1 (DE):** {brief.get('h1_german')}")
                                    st.markdown(f"**H1 (EN):** {brief.get('h1_english')}")
                                    st.markdown("**Outline:**")
                                    for sec in brief.get('outline', []):
                                        st.markdown(f"- **{sec.get('h2')}** ({sec.get('intent')})")
        
        with tab_data:
            st.dataframe(df_clustered, use_container_width=True)
            csv = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button("Download Strategy CSV", csv, "german_clusters.csv", "text/csv")
            
    else:
        st.warning("No keywords found.")

elif run_btn:
    st.error("API Key required.")
