import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime, timedelta
import google.generativeai as genai
import plotly.express as px
import re

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION (Dejan Style)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Cross-Border Keyword Bridge", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    
    h1, h2, h3 { color: #111; font-weight: 600; letter-spacing: -0.5px; }
    
    .metric-card {
        background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 10px;
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stDataFrame"] { border: 1px solid #e1e4e8; }
    
    .tech-note { font-size: 0.85rem; color: #57606a; background-color: #f6f8fa; border-left: 3px solid #0969da; padding: 12px; border-radius: 0 4px 4px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC ENGINE (Dynamic Discovery)
# -----------------------------------------------------------------------------

def get_active_model_name(api_key):
    """
    Connects to Google, lists available models, and picks the best text generator.
    """
    genai.configure(api_key=api_key)
    try:
        # Get all models available to this key
        all_models = list(genai.list_models())
        
        # Filter for models that can generate content
        text_models = [m for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not text_models:
            return "gemini-1.5-flash" # Desperate fallback
            
        # Priority 1: Flash (Fastest/Cheapest)
        for m in text_models:
            if 'flash' in m.name.lower(): return m.name
            
        # Priority 2: Pro
        for m in text_models:
            if 'pro' in m.name.lower(): return m.name
            
        # Priority 3: First available
        return text_models[0].name
        
    except Exception as e:
        # If listing fails, guess the most common one
        return "gemini-1.5-flash"

def run_gemini_prompt(api_key, prompt):
    model_name = get_active_model_name(api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Clean output
        text = response.text
        if "```" in text:
            text = re.sub(r"```json|```", "", text).strip()
            
        return json.loads(text)
        
    except Exception as e:
        # Raise specific error for debugging
        raise Exception(f"Model ({model_name}) failed: {str(e)}")

def get_synonyms_strategy(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term.
    2. The formal/medical term.
    3. A popular synonym.
    
    Return raw JSON:
    {{
        "synonyms": ["term1", "term2", "term3"],
        "explanation": "Brief reasoning"
    }}
    """
    try:
        return run_gemini_prompt(api_key, prompt)
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None

def batch_translate_full(api_key, all_keywords):
    if not all_keywords: return {}
    
    full_map = {}
    # Chunk size 50
    chunks = [all_keywords[i:i + 50] for i in range(0, len(all_keywords), 50)]
    
    prog_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        prog_text.text(f"Translating batch {i+1}/{len(chunks)}...")
        
        prompt = f"""
        Translate these German keywords to English. Keep it literal.
        Input: {json.dumps(chunk)}
        Return JSON: {{ "German Keyword": "English Translation" }}
        """
        try:
            res = run_gemini_prompt(api_key, prompt)
            if res: full_map.update(res)
            time.sleep(0.2)
        except:
            continue
        
    prog_text.empty()
    return full_map

def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine_synonyms(synonyms):
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kosten", " tipps", " kaufen"]
    all_data = []
    
    p_bar = st.progress(0, text="Mining Google...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            p_bar.progress(min(step / total, 1.0), text=f"Mining: '{seed}{mod}'...")
            
            results = fetch_suggestions(f"{seed}{mod}")
            
            intent = "General"
            if "für" in mod: intent = "Use Case"
            elif "gegen" in mod: intent = "Solution"
            elif "hausmittel" in mod: intent = "DIY"
            elif "kaufen" in mod: intent = "Transactional"
            
            for r in results:
                all_data.append({"German Keyword": r, "Seed Term": seed, "Intent": intent})
            
            time.sleep(0.1)
            
    p_bar.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['German Keyword'], keep='first')
    return df

# --- WIKIPEDIA TRAFFIC PROXY ---
def get_wiki_traffic(german_keyword):
    headers = {'User-Agent': 'SEOTool/1.0'}
    # 1. Resolve Entity
    search_url = f"https://de.wikipedia.org/w/api.php?action=opensearch&search={german_keyword}&limit=1&namespace=0&format=json"
    try:
        s_resp = requests.get(search_url, headers=headers, timeout=2)
        if s_resp.status_code == 200:
            results = s_resp.json()
            if results[1]:
                article = results[1][0]
                safe_title = article.replace(" ", "_")
                
                # 2. Get Stats
                end = datetime.now()
                start = end - timedelta(days=30)
                metrics_url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/de.wikipedia/all-access/user/{safe_title}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
                
                m_resp = requests.get(metrics_url, headers=headers, timeout=2)
                if m_resp.status_code == 200:
                    data = m_resp.json()
                    total = sum([i['views'] for i in data.get('items', [])])
                    return total, article
    except: pass
    return 0, "-"

def batch_fetch_wiki_scores(keywords_list):
    targets = keywords_list[:30] # Top 30 to keep fast
    scores = {}
    entities = {}
    prog_text = st.empty()
    
    for i, kw in enumerate(targets):
        prog_text.text(f"Checking Wiki Interest: {kw}...")
        views, entity = get_wiki_traffic(kw)
        scores[kw] = views
        entities[kw] = entity
        time.sleep(0.05)
        
    prog_text.empty()
    return scores, entities

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Config")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("""<a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;color:#0969da;">🔑 Get Free Key</a>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="tech-note">
    <b>Wikipedia Proxy:</b> 
    We use German Wikipedia Pageviews as a proxy for "Informational Interest." 
    <br>High Wiki traffic often correlates with high Google search volume for informational queries.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------

st.title("Cross-Border Keyword Bridge 🇩🇪")
st.markdown("### German Long-Tail Intelligence")

keyword = st.text_input("Enter English Keyword", placeholder="e.g. newborn babies")
run_btn = st.button("Generate German Strategy", type="primary")

if run_btn and keyword and api_key:
    
    # 1. AI Strategy
    with st.spinner("Consulting German Lexicon..."):
        strategy = get_synonyms_strategy(api_key, keyword)
    
    if strategy:
        synonyms = strategy.get('synonyms', [])
        
        # 2. Deep Mining
        df_keywords = deep_mine_synonyms(synonyms)
        
        # 3. Translation & Metrics
        if not df_keywords.empty:
            # Translate
            with st.spinner(f"Translating keywords..."):
                germ_list = df_keywords['German Keyword'].tolist()
                translations = batch_translate_full(api_key, germ_list)
                df_keywords['English Meaning'] = df_keywords['German Keyword'].map(translations).fillna("-")

            # Wiki Scores
            sorted_keywords = sorted(germ_list, key=len)
            scores, entities = batch_fetch_wiki_scores(sorted_keywords)
            
            df_keywords['Monthly Wiki Views'] = df_keywords['German Keyword'].map(scores).fillna(0).astype(int)
            df_keywords['Wiki Topic'] = df_keywords['German Keyword'].map(entities).fillna("-")

        # --- OUTPUT ---
        st.markdown("---")
        st.markdown(f"### 1. Strategic Translation")
        st.info(f"**AI Insight:** {strategy.get('explanation', '')}")
        
        cols = st.columns(len(synonyms))
        for i, syn in enumerate(synonyms):
            cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader(f"2. The Master Matrix")
        st.caption("Sorted by Wiki Interest (Informational Demand Proxy)")
        
        if not df_keywords.empty:
            # Filter
            all_intents = ["All"] + list(df_keywords['Intent'].unique())
            sel_intent = st.selectbox("Filter by Intent:", all_intents)
            
            df_display = df_keywords if sel_intent == "All" else df_keywords[df_keywords['Intent'] == sel_intent]
            
            # Sort by Wiki Views descending
            df_display = df_display.sort_values('Monthly Wiki Views', ascending=False)
            
            # Reorder
            df_display = df_display[['German Keyword', 'English Meaning', 'Monthly Wiki Views', 'Wiki Topic', 'Intent', 'Seed Term']]

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "German Keyword": st.column_config.TextColumn("🇩🇪 German Query", width="medium"),
                    "English Meaning": st.column_config.TextColumn("🇺🇸 English Meaning", width="medium"),
                    "Monthly Wiki Views": st.column_config.ProgressColumn(
                        "Wiki Interest", 
                        help="Pageviews for the associated Wikipedia article (Last 30 Days). High views = High informational demand.",
                        format="%d", 
                        min_value=0, 
                        max_value=int(df_display['Monthly Wiki Views'].max())
                    ),
                    "Mapped Entity": st.column_config.TextColumn("Wiki Topic", width="small"),
                }
            )
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Master List (CSV)", csv, "german_strategy.csv", "text/csv")
        else:
            st.warning("No keywords found.")
        
    else:
        st.error("AI Analysis Failed. Check API Key.")

elif run_btn:
    st.error("Enter API Key and Keyword.")
