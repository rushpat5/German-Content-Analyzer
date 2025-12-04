import streamlit as st
import pandas as pd
import requests
import json
import time
import google.generativeai as genai
from pytrends.request import TrendReq
import plotly.express as px
import re

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Cross-Border Keyword Bridge", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    h1, h2, h3 { color: #111; font-weight: 600; letter-spacing: -0.5px; }
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 10px; }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stDataFrame"] { border: 1px solid #e1e4e8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC ENGINE
# -----------------------------------------------------------------------------

def get_valid_model(api_key):
    genai.configure(api_key=api_key)
    try:
        models = list(genai.list_models())
        valid = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        for m in valid:
            if 'flash' in m and '1.5' in m: return m
        return "models/gemini-pro"
    except:
        return "models/gemini-1.5-flash"

def get_cultural_translation(api_key, keyword):
    try:
        model_name = get_valid_model(api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        Act as a Native German SEO Expert.
        English Keyword: "{keyword}"
        Task: Identify the top 3 distinct German terms used for this concept. 
        Return ONLY valid JSON. No Markdown.
        {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Reasoning" }}
        """
        resp = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        return json.loads(text)
    except Exception as e:
        st.error(f"Translation Error: {str(e)}")
        return None

def batch_translate_full(api_key, all_keywords):
    if not all_keywords: return {}
    model_name = get_valid_model(api_key)
    model = genai.GenerativeModel(model_name)
    full_map = {}
    chunks = [all_keywords[i:i + 50] for i in range(0, len(all_keywords), 50)]
    
    prog_text = st.empty()
    for i, chunk in enumerate(chunks):
        prog_text.text(f"Translating batch {i+1}/{len(chunks)}...")
        try:
            prompt = f"""
            Translate these German search queries into English. Keep it literal and short.
            Input: {json.dumps(chunk)}
            Return ONLY raw JSON key-value pairs: {{ "German Keyword": "English Translation" }}
            """
            resp = model.generate_content(prompt)
            text = re.sub(r"```json|```", "", resp.text).strip()
            full_map.update(json.loads(text))
            time.sleep(0.5)
        except: continue
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
    
    prog_bar = st.progress(0, text="Mining Google.de...")
    total_steps = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            prog_bar.progress(min(step / total_steps, 1.0))
            
            query = f"{seed}{mod}"
            results = fetch_suggestions(query)
            
            intent = "General"
            if "für" in mod: intent = "Use Case"
            elif "gegen" in mod: intent = "Solution"
            elif "hausmittel" in mod: intent = "DIY"
            elif "kaufen" in mod: intent = "Transactional"
            
            for r in results:
                all_data.append({"German Keyword": r, "Seed Term": seed, "Intent": intent})
            time.sleep(0.1)
            
    prog_bar.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['German Keyword'], keep='first')
    return df

def get_google_trends(keywords):
    """Base trend fetcher for charts."""
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        pytrends.build_payload(keywords[:1], cat=0, timeframe='today 12-m', geo='DE')
        data = pytrends.interest_over_time()
        if not data.empty: return data.drop(columns=['isPartial'])
    except: return None
    return None

def batch_fetch_trend_scores(keywords_list):
    """
    Fetches the LATEST trend score (0-100) for a list of keywords.
    Batches in groups of 5 to respect Google limits.
    """
    pytrends = TrendReq(hl='de-DE', tz=360)
    trend_map = {}
    
    # We limit to Top 20 to prevent Timeouts/IP Bans in this demo environment
    # In a production environment, you would use proxies or delays.
    target_list = keywords_list[:20] 
    
    chunks = [target_list[i:i + 5] for i in range(0, len(target_list), 5)]
    
    prog_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        prog_text.text(f"Fetching Trends for batch {i+1}/{len(chunks)}...")
        try:
            pytrends.build_payload(chunk, cat=0, timeframe='now 1-H', geo='DE') # Last 1 hour for "Current" or 'today 1-m'
            data = pytrends.interest_over_time()
            
            if not data.empty:
                # Get the last row (most recent data point)
                latest_values = data.iloc[-1]
                for kw in chunk:
                    if kw in latest_values:
                        val = latest_values[kw]
                        trend_map[kw] = int(val)
            time.sleep(1.5) # Critical wait to avoid 429
        except:
            continue
            
    prog_text.empty()
    return trend_map

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Config")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("""<a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;color:#0969da;">🔑 Get Free Key</a>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Note: Trend Scores (0-100) are fetched for the Top 20 keywords to prevent Google API rate limiting.")

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------

st.title("Cross-Border Keyword Bridge 🇩🇪")
st.markdown("### German Long-Tail Intelligence")

keyword = st.text_input("Enter English Keyword", placeholder="e.g. newborn babies")
run_btn = st.button("Generate German Strategy", type="primary")

if run_btn and keyword and api_key:
    
    # 1. Synonyms
    with st.spinner("Consulting German Lexicon..."):
        strategy = get_cultural_translation(api_key, keyword)
    
    if strategy:
        synonyms = strategy.get('synonyms', [])
        
        # 2. Mining
        df_keywords = deep_mine_synonyms(synonyms)
        
        # 3. Translation
        if not df_keywords.empty:
            with st.spinner(f"Translating keywords..."):
                germ_list = df_keywords['German Keyword'].tolist()
                translations = batch_translate_full(api_key, germ_list)
                df_keywords['English Meaning'] = df_keywords['German Keyword'].map(translations).fillna("-")

            # 4. TREND FETCHING (NEW)
            # We sort by length to prioritize shorter (likely higher volume) terms first for the trend check
            sorted_keywords = sorted(germ_list, key=len)
            trend_scores = batch_fetch_trend_scores(sorted_keywords)
            
            # Map trends to dataframe
            df_keywords['Current Trend (0-100)'] = df_keywords['German Keyword'].map(trend_scores).fillna("N/A")

        # 5. Main Trend Chart
        trend_data = get_google_trends([synonyms[0]])

        # --- OUTPUT ---
        st.markdown("---")
        st.markdown(f"### 1. Strategic Translation")
        st.info(f"**AI Insight:** {strategy.get('explanation', '')}")
        
        cols = st.columns(len(synonyms))
        for i, syn in enumerate(synonyms):
            cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
            
        st.markdown("---")
        st.subheader("2. Demand Trend")
        if trend_data is not None:
            fig = px.line(trend_data, y=synonyms[0], title=f"Interest: '{synonyms[0]}'", color_discrete_sequence=['#1a7f37'])
            fig.update_layout(plot_bgcolor='white', yaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Trend unavailable.")

        st.markdown("---")
        st.subheader(f"3. The Master Matrix")
        
        if not df_keywords.empty:
            all_intents = ["All"] + list(df_keywords['Intent'].unique())
            sel_intent = st.selectbox("Filter by Intent:", all_intents)
            
            df_display = df_keywords if sel_intent == "All" else df_keywords[df_keywords['Intent'] == sel_intent]
            
            # Reorder for best view
            df_display = df_display[['German Keyword', 'English Meaning', 'Current Trend (0-100)', 'Intent', 'Seed Term']]

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "German Keyword": st.column_config.TextColumn("🇩🇪 German Query", width="medium"),
                    "English Meaning": st.column_config.TextColumn("🇺🇸 English Meaning", width="medium"),
                    "Current Trend (0-100)": st.column_config.TextColumn("🔥 Interest", help="Relative search volume (0-100) in the last hour. N/A means too low or rate limited."),
                }
            )
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Master List (CSV)", csv, "german_strategy.csv", "text/csv")
        else:
            st.warning("No keywords found.")
        
    else:
        st.error("AI Analysis Failed.")

elif run_btn:
    st.error("Enter API Key and Keyword.")
