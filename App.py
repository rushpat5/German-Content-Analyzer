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
# 1. VISUAL CONFIGURATION (Dejan Style)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Cross-Border Keyword Bridge", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    
    h1, h2, h3 { color: #111; font-weight: 600; letter-spacing: -0.5px; }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 10px;
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Sidebar */
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
        return "models/gemini-pro"

def get_synonyms_strategy(api_key, keyword):
    """
    Asks AI for ALL valid German variations (Synonyms) to ensure we don't miss traffic.
    """
    try:
        model_name = get_valid_model(api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        Act as a German Native SEO.
        English Keyword: "{keyword}"
        
        Task: Identify the top 3 distinct German terms used for this concept. 
        Include colloquial terms, medical terms, or synonyms if they are searched.
        
        Example for "Diaper Rash": ["Wunder Po", "Windeldermatitis", "Wundsein"]
        Example for "Cell Phone": ["Handy", "Smartphone", "Mobiltelefon"]
        
        Return raw JSON:
        {{
            "synonyms": ["term1", "term2", "term3"],
            "explanation": "Brief reason for these choices"
        }}
        """
        resp = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        return json.loads(text)
    except:
        return None

def batch_translate_to_english(api_key, german_keywords):
    """
    Translates a list of German keywords back to English for the user.
    """
    if not german_keywords: return {}
    
    try:
        model_name = get_valid_model(api_key)
        model = genai.GenerativeModel(model_name)
        
        # We process top 50 to save time/tokens, or all if small
        targets = german_keywords[:80]
        
        prompt = f"""
        Translate these German search queries into English. Keep it short.
        Input List: {json.dumps(targets)}
        
        Return raw JSON key-value pairs:
        {{
            "German Keyword": "English Translation"
        }}
        """
        resp = model.generate_content(prompt)
        text = re.sub(r"```json|```", "", resp.text).strip()
        return json.loads(text)
    except:
        return {}

def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine_synonyms(synonyms):
    """
    Runs the Recursive Mining on MULTIPLE synonyms (Multi-Seed Mining).
    """
    # German modifiers to force long-tail
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kosten", " tipps"]
    
    all_data = []
    
    progress_bar = st.progress(0, text="Initializing Mining...")
    total_steps = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            progress_bar.progress(step / total_steps, text=f"Mining Google.de: '{seed}{mod}'...")
            
            query = f"{seed}{mod}"
            results = fetch_suggestions(query)
            
            intent = "General"
            if "für" in mod: intent = "Use Case"
            elif "gegen" in mod: intent = "Problem Solving"
            elif "hausmittel" in mod: intent = "DIY / Remedies"
            elif "wann" in mod: intent = "Informational"
            
            for r in results:
                all_data.append({
                    "German Keyword": r,
                    "Seed Term": seed,
                    "Intent": intent
                })
            
            time.sleep(0.05) # Rate limit safety
            
    progress_bar.empty()
    
    # Deduplicate
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['German Keyword'])
    return df

def get_trend(keyword):
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='DE')
        data = pytrends.interest_over_time()
        if not data.empty: return data.drop(columns=['isPartial'])
    except: return None
    return None

# -----------------------------------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Config")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.markdown("""
    <div class="tech-note">
    <b>Multi-Seed Mining:</b> 
    We don't just translate once. We find the top 3 German variations (e.g. <i>Baby, Säugling, Neugeborenes</i>) and mine Google Autocomplete for ALL of them to ensure 100% coverage.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------

st.title("Cross-Border Keyword Bridge 🇩🇪")
st.markdown("### German Long-Tail Intelligence & Strategy")

keyword = st.text_input("Enter English Keyword", placeholder="e.g. newborn babies")
run_btn = st.button("Generate German Strategy", type="primary")

if run_btn and keyword and api_key:
    
    # 1. Identify Synonyms
    with st.spinner("Consulting German Lexicon..."):
        strategy = get_synonyms_strategy(api_key, keyword)
    
    if strategy:
        synonyms = strategy['synonyms']
        
        # 2. Deep Mine (Multi-Seed)
        df_keywords = deep_mine_synonyms(synonyms)
        
        # 3. Back-Translate Top Results
        with st.spinner(f"Translating {len(df_keywords)} keywords back to English for you..."):
            # Extract list of german words to translate
            germ_list = df_keywords['German Keyword'].tolist()
            translations = batch_translate_to_english(api_key, germ_list)
            
            # Map back to dataframe
            df_keywords['English Meaning'] = df_keywords['German Keyword'].map(translations).fillna("-")

        # 4. Get Trends for the main synonym
        main_term = synonyms[0]
        trend_data = get_trend(main_term)

        # --- OUTPUT ---
        st.markdown("---")
        
        # A. Strategy Header
        st.markdown(f"### 1. Strategy: {keyword}")
        st.markdown(f"**AI Insight:** {strategy['explanation']}")
        
        cols = st.columns(len(synonyms))
        for i, syn in enumerate(synonyms):
            cols[i].markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{syn}</div>
                <div class="metric-lbl">German Variation #{i+1}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # B. Trend
        st.markdown("---")
        st.subheader("2. Seasonality (Google Trends)")
        if trend_data is not None:
            fig = px.line(trend_data, y=main_term, title=f"Interest in '{main_term}' (Germany)", color_discrete_sequence=['#1a7f37'])
            fig.update_layout(plot_bgcolor='white', yaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trend data unavailable for this specific term.")

        # C. The Master Matrix
        st.markdown("---")
        st.subheader(f"3. The Master Matrix ({len(df_keywords)} Keywords)")
        st.markdown("We combined results from all German variations into one list, translated back to English for you.")
        
        # Reorder columns for readability
        df_display = df_keywords[['German Keyword', 'English Meaning', 'Intent', 'Seed Term']]
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "German Keyword": st.column_config.TextColumn("🇩🇪 German Query", width="medium"),
                "English Meaning": st.column_config.TextColumn("🇺🇸 English Meaning", width="medium"),
                "Seed Term": st.column_config.TextColumn("Source Synonym", width="small"),
            }
        )
        
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button("Download Master List (CSV)", csv, "german_strategy.csv", "text/csv")
        
    else:
        st.error("AI Analysis Failed. Please check API Key.")

elif run_btn:
    st.error("Enter API Key and Keyword.")
