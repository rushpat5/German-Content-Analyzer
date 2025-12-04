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
# 2. LOGIC ENGINE (Dynamic & Robust)
# -----------------------------------------------------------------------------

def get_best_model_name(api_key):
    """
    Dynamically asks Google which models are available to this Key
    and selects the best one for text generation.
    """
    genai.configure(api_key=api_key)
    try:
        # 1. List models
        models = list(genai.list_models())
        
        # 2. Filter for text generation capability
        valid_models = [m for m in models if 'generateContent' in m.supported_generation_methods]
        
        # 3. Priority Selection Strategy
        # Priority A: Flash 1.5 (Fastest/Cheapest)
        for m in valid_models:
            if 'flash' in m.name and '1.5' in m.name: return m.name
            
        # Priority B: Pro 1.5 (Smartest)
        for m in valid_models:
            if 'pro' in m.name and '1.5' in m.name: return m.name
            
        # Priority C: Any Gemini
        for m in valid_models:
            if 'gemini' in m.name: return m.name
            
        # Fallback if list is empty but auth passed (rare)
        return "models/gemini-1.5-flash"
        
    except Exception as e:
        # If listing fails (often due to API key permissions), force a standard default
        return "models/gemini-1.5-flash"

def run_gemini_prompt(api_key, prompt):
    try:
        # Dynamically find the working model name
        model_name = get_best_model_name(api_key)
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        text = response.text
        if "```" in text:
            text = re.sub(r"```json|```", "", text).strip()
            
        return json.loads(text)
    except Exception as e:
        # Raise error to be caught by caller
        raise Exception(f"Model Error ({model_name}): {str(e)}")

def get_synonyms_strategy(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term (what real people type).
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
    
    # Chunk size 40 to be safe
    chunks = [all_keywords[i:i + 40] for i in range(0, len(all_keywords), 40)]
    
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
            full_map.update(res)
        except: continue
        time.sleep(0.5)
        
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
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kosten", " kaufen"]
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
            elif "wann" in mod or "was" in mod: intent = "Informational"
            
            for r in results:
                all_data.append({
                    "German Keyword": r,
                    "Seed Term": seed,
                    "Intent": intent,
                    "Length": len(r) # For sorting priority
                })
            time.sleep(0.1)
            
    p_bar.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['German Keyword'], keep='first')
    return df

# --- SMART TREND FETCHING ---
def fetch_smart_trends(df_keywords):
    """
    Selects the Top 15 most relevant keywords and fetches their Google Trends score.
    Strategy: Shortest words + High Intent words.
    """
    # 1. Prioritize: Sort by Length (Shortest first = Head Terms)
    candidates = df_keywords.sort_values('Length', ascending=True)
    
    # 2. Take Top 15 to avoid Rate Limits
    target_list = candidates['German Keyword'].head(15).tolist()
    
    trend_map = {}
    pytrends = TrendReq(hl='de-DE', tz=360)
    
    # Chunk into batches of 5 (Google Limit)
    batches = [target_list[i:i + 5] for i in range(0, len(target_list), 5)]
    
    prog_text = st.empty()
    
    for i, batch in enumerate(batches):
        prog_text.text(f"Checking Google Trends Batch {i+1}/{len(batches)}...")
        try:
            # Fetch last 3 months
            pytrends.build_payload(batch, cat=0, timeframe='today 3-m', geo='DE')
            time.sleep(1.5) # Safety pause
            data = pytrends.interest_over_time()
            
            if not data.empty:
                # Calculate Average Score over last 3 months
                means = data.mean()
                for kw in batch:
                    if kw in means:
                        trend_map[kw] = int(means[kw])
        except:
            continue
            
    prog_text.empty()
    return trend_map

def get_google_trends_single(keyword):
    """Fetches trend for the main chart."""
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='DE')
        time.sleep(0.5)
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
    st.markdown("""<a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;color:#0969da;">🔑 Get Free Key</a>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="tech-note">
    <b>Google Trends Smart-Batching:</b> 
    To avoid API bans (Error 429), we intelligently select the <b>Top 15 High-Potential Keywords</b> (based on brevity and intent) and fetch their relative search volume (0-100).
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
        
        if not df_keywords.empty:
            # 3. Translation
            with st.spinner(f"Translating keywords..."):
                germ_list = df_keywords['German Keyword'].tolist()
                translations = batch_translate_full(api_key, germ_list)
                df_keywords['English Meaning'] = df_keywords['German Keyword'].map(translations).fillna("-")

            # 4. SMART TRENDS (Top 15 Only)
            scores = fetch_smart_trends(df_keywords)
            
            # Map scores (Default to -1 if not fetched to sort them to bottom)
            df_keywords['Trend Score (3mo)'] = df_keywords['German Keyword'].map(scores).fillna(-1).astype(int)
            
            # Clean up display (Replace -1 with "N/A")
            df_keywords['Trend Display'] = df_keywords['Trend Score (3mo)'].apply(lambda x: str(x) if x >= 0 else "N/A (Low Prio)")

            # --- OUTPUT ---
            st.markdown("---")
            st.markdown(f"### 1. Strategic Translation")
            st.info(f"**AI Insight:** {strategy.get('explanation', '')}")
            
            cols = st.columns(len(synonyms))
            for i, syn in enumerate(synonyms):
                cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
            
            # 5. Main Trend Chart
            st.markdown("---")
            st.subheader(f"2. Demand Trend: '{synonyms[0]}'")
            trend_data = get_google_trends_single(synonyms[0])
            if trend_data is not None:
                fig = px.line(trend_data, y=synonyms[0], title=f"Relative Interest (Last 12 Months)", color_discrete_sequence=['#1a7f37'])
                fig.update_layout(plot_bgcolor='white', yaxis=dict(gridcolor='#f0f0f0'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Trend data unavailable (Google Trends API limit or low volume).")

            st.markdown("---")
            st.subheader(f"3. The Master Matrix")
            st.caption("Top Keywords checked against Google Trends (0-100 Scale). 'N/A' means the keyword was too long-tail to prioritize for the trend check.")
            
            # Sort: High Trend Score first, then Shortest words (likely highest volume)
            df_display = df_keywords.sort_values(by=['Trend Score (3mo)', 'Length'], ascending=[False, True])
            
            df_display = df_display[['German Keyword', 'English Meaning', 'Trend Display', 'Intent', 'Seed Term']]

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "German Keyword": st.column_config.TextColumn("🇩🇪 German Query", width="medium"),
                    "English Meaning": st.column_config.TextColumn("🇺🇸 English Meaning", width="medium"),
                    "Trend Display": st.column_config.TextColumn("🔥 Google Trend (0-100)"),
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
