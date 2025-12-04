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
    .tech-note { font-size: 0.85rem; color: #57606a; background-color: #f6f8fa; border-left: 3px solid #0969da; padding: 12px; border-radius: 0 4px 4px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC ENGINE
# -----------------------------------------------------------------------------

def get_best_model_name(api_key):
    genai.configure(api_key=api_key)
    try:
        all_models = list(genai.list_models())
        text_models = [m for m in all_models if 'generateContent' in m.supported_generation_methods]
        if not text_models: return "models/gemini-1.5-flash"
        for m in text_models:
            if 'flash' in m.name.lower(): return m.name
        return text_models[0].name
    except: return "models/gemini-1.5-flash"

def run_gemini_prompt(api_key, prompt):
    model_name = get_best_model_name(api_key)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text
        if "```" in text: text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except Exception as e:
        raise Exception(f"Model Error: {str(e)}")

def get_synonyms_strategy(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Topic: "{keyword}"
    
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term.
    2. The formal/medical term.
    3. A popular synonym.
    
    Return raw JSON: {{ "synonyms": ["term1", "term2", "term3"], "explanation": "Reasoning" }}
    """
    try: return run_gemini_prompt(api_key, prompt)
    except: return None

def batch_validate_and_translate(api_key, all_keywords, original_topic):
    """
    Translates AND filters out junk (Brands, Irrelevant topics).
    """
    if not all_keywords: return {}
    full_map = {}
    
    # Chunk size 30 (smaller chunks for complex logic)
    chunks = [all_keywords[i:i + 30] for i in range(0, len(all_keywords), 30)]
    
    prog_text = st.empty()
    for i, chunk in enumerate(chunks):
        prog_text.text(f"AI Filtering & Translating batch {i+1}/{len(chunks)}...")
        
        prompt = f"""
        Context: I am researching the topic "{original_topic}" for the German market.
        I have a list of scraped keywords. 
        
        Task: 
        1. Translate the German keyword to English.
        2. Mark as "RELEVANT" only if it relates to "{original_topic}".
        3. Mark as "IRRELEVANT" if it is a specific Brand Name (like 'BabyOne', 'DM', 'Rossmann'), a Store, a Movie, or a totally different topic (e.g. 'Babyboomer').
        
        Input List: {json.dumps(chunk)}
        
        Return JSON Object: 
        {{
            "german_keyword_1": {{ "en": "translation", "keep": true }},
            "german_keyword_2": {{ "en": "translation", "keep": false }}
        }}
        """
        try:
            res = run_gemini_prompt(api_key, prompt)
            full_map.update(res)
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
            
            for r in results:
                all_data.append({
                    "German Keyword": r,
                    "Seed Term": seed,
                    "Intent": intent,
                    "Length": len(r)
                })
            time.sleep(0.1)
            
    p_bar.empty()
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['German Keyword'], keep='first')
    return df

def fetch_smart_trends(df_keywords):
    # Prioritize Short + Relevant keywords
    candidates = df_keywords.sort_values('Length', ascending=True)
    target_list = candidates['German Keyword'].head(15).tolist()
    trend_map = {}
    pytrends = TrendReq(hl='de-DE', tz=360)
    batches = [target_list[i:i + 5] for i in range(0, len(target_list), 5)]
    
    prog_text = st.empty()
    for i, batch in enumerate(batches):
        prog_text.text(f"Checking Trends Batch {i+1}...")
        try:
            pytrends.build_payload(batch, cat=0, timeframe='today 3-m', geo='DE')
            time.sleep(1.5)
            data = pytrends.interest_over_time()
            if not data.empty:
                means = data.mean()
                for kw in batch:
                    if kw in means: trend_map[kw] = int(means[kw])
        except: continue
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
    st.markdown("""
    <div class="tech-note">
    <b>Smart Filter:</b> 
    The AI now validates every scraped keyword against your original topic ("{keyword}"). It automatically discards irrelevant brands (e.g. <i>Babyone</i>) and off-topic homonyms (e.g. <i>Babylon</i>).
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
            # 3. VALIDATION & TRANSLATION (New Logic)
            with st.spinner(f"Validating & Translating keywords..."):
                germ_list = df_keywords['German Keyword'].tolist()
                # Pass the original user keyword to context
                validation_map = batch_validate_and_translate(api_key, germ_list, keyword)
                
                # Apply data from map
                df_keywords['English Meaning'] = df_keywords['German Keyword'].apply(lambda x: validation_map.get(x, {}).get('en', '-'))
                df_keywords['Keep'] = df_keywords['German Keyword'].apply(lambda x: validation_map.get(x, {}).get('keep', False))
            
            # FILTER OUT JUNK
            df_clean = df_keywords[df_keywords['Keep'] == True].copy()
            
            if df_clean.empty:
                st.warning("All mined keywords were flagged as irrelevant brands/noise. Showing raw list instead.")
                df_clean = df_keywords # Fallback if AI deletes everything
            else:
                st.success(f"Filtered out {len(df_keywords) - len(df_clean)} irrelevant/brand terms (e.g. Babyone, Babylon).")

            # 4. SMART TRENDS
            scores = fetch_smart_trends(df_clean)
            df_clean['Trend Score'] = df_clean['German Keyword'].map(scores).fillna(-1).astype(int)
            df_clean['Trend Display'] = df_clean['Trend Score'].apply(lambda x: str(x) if x >= 0 else "N/A")

            # --- OUTPUT ---
            st.markdown("---")
            st.markdown(f"### 1. Strategic Translation")
            st.info(f"**AI Insight:** {strategy.get('explanation', '')}")
            
            cols = st.columns(len(synonyms))
            for i, syn in enumerate(synonyms):
                cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader(f"2. The Master Matrix")
            st.caption("Cleaned, translated, and trend-checked.")
            
            df_display = df_clean.sort_values(by=['Trend Score', 'Length'], ascending=[False, True])
            df_display = df_display[['German Keyword', 'English Meaning', 'Trend Display', 'Intent', 'Seed Term']]

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "German Keyword": st.column_config.TextColumn("🇩🇪 German Query", width="medium"),
                    "English Meaning": st.column_config.TextColumn("🇺🇸 English Meaning", width="medium"),
                    "Trend Display": st.column_config.TextColumn("🔥 Trend (0-100)"),
                }
            )
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clean List (CSV)", csv, "german_strategy.csv", "text/csv")
        else:
            st.warning("No keywords found.")
        
    else:
        st.error("AI Analysis Failed. Check API Key.")

elif run_btn:
    st.error("Enter API Key and Keyword.")
