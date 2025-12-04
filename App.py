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
    /* --- FORCE LIGHT MODE --- */
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
    section[data-testid="stSidebar"] * { color: #24292e !important; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stDataFrame"] { border: 1px solid #e1e4e8; }
    
    /* Tech Note */
    .tech-note { font-size: 0.85rem; color: #57606a; background-color: #f6f8fa; border-left: 3px solid #0969da; padding: 12px; border-radius: 0 4px 4px 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC ENGINE
# -----------------------------------------------------------------------------

def get_best_available_model(api_key):
    """
    Dynamically asks Google which models are available to this Key
    and picks the best one (Flash > Pro > Others).
    """
    genai.configure(api_key=api_key)
    try:
        # 1. List all models the key can access
        all_models = list(genai.list_models())
        
        # 2. Filter for text generation models
        text_models = [m for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        # 3. Priority Selection
        # Try to find Flash first (Fast/Free)
        for m in text_models:
            if 'flash' in m.name.lower() and '1.5' in m.name:
                return m.name
        
        # Try Pro 1.5
        for m in text_models:
            if 'pro' in m.name.lower() and '1.5' in m.name:
                return m.name
                
        # Fallback to any Gemini model found
        for m in text_models:
            if 'gemini' in m.name.lower():
                return m.name
                
        # Ultimate Fallback (Blind Guess)
        return "models/gemini-1.5-flash"
        
    except Exception as e:
        # If list_models fails (permissions), default to Flash
        return "models/gemini-1.5-flash"

def run_gemini_prompt(api_key, prompt):
    # Find the working model name dynamically
    model_name = get_best_available_model(api_key)
    # st.toast(f"Using Model: {model_name}") # Optional debug
    
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Clean output
        text = response.text
        if "```" in text:
            text = re.sub(r"```json|```", "", text).strip()
            
        return json.loads(text)
        
    except Exception as e:
        raise Exception(f"Model ({model_name}) Failed: {str(e)}")

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term (what real people type).
    2. The formal/medical/technical term (if applicable).
    3. A popular synonym or related concept.
    
    Return ONLY valid JSON. No Markdown.
    Format:
    {{
        "synonyms": ["term1", "term2", "term3"],
        "explanation": "Brief reason for these choices"
    }}
    """
    try:
        return run_gemini_prompt(api_key, prompt)
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return None

def batch_translate_to_english(api_key, german_keywords):
    if not german_keywords: return {}
    
    # Chunking to avoid token limits (max 50 at a time)
    targets = german_keywords[:50] 
    
    prompt = f"""
    Translate these German search queries into English. Keep it short and literal.
    Input List: {json.dumps(targets)}
    
    Return ONLY raw JSON key-value pairs:
    {{
        "German Keyword": "English Translation"
    }}
    """
    try:
        return run_gemini_prompt(api_key, prompt)
    except:
        return {}

def fetch_suggestions(query):
    """Safe Google Autocomplete Scraper"""
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine_synonyms(synonyms):
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kosten", " tipps", " kaufen"]
    all_data = []
    
    progress_bar = st.progress(0, text="Initializing Mining...")
    total_steps = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            progress_bar.progress(min(step / total_steps, 1.0), text=f"Mining Google.de: '{seed}{mod}'...")
            
            query = f"{seed}{mod}"
            results = fetch_suggestions(query)
            
            # Determine Intent based on Modifier
            intent = "General"
            if "für" in mod: intent = "Use Case"
            elif "gegen" in mod: intent = "Solution / Remedy"
            elif "hausmittel" in mod: intent = "Informational (DIY)"
            elif "wann" in mod or "was" in mod: intent = "Informational (Q&A)"
            elif "kaufen" in mod or "kosten" in mod: intent = "Transactional"
            elif "tipps" in mod: intent = "Informational"
            
            for r in results:
                all_data.append({
                    "German Keyword": r,
                    "Seed Term": seed,
                    "Intent": intent,
                    "Modifier Used": mod.strip() if mod else "Head Term"
                })
            
            time.sleep(0.1) # Respect Google
            
    progress_bar.empty()
    
    # Deduplicate
    df = pd.DataFrame(all_data)
    if not df.empty:
        df = df.drop_duplicates(subset=['German Keyword'])
    return df

def get_google_trends(keywords):
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        target = keywords[:1] 
        pytrends.build_payload(target, cat=0, timeframe='today 12-m', geo='DE', gprop='')
        time.sleep(0.5)
        data = pytrends.interest_over_time()
        if not data.empty:
            return data.drop(columns=['isPartial'])
    except:
        return None
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
    <b>Methodology:</b>
    <br>1. <b>Synonym Generation:</b> Auto-detects 3 variations (Colloquial, Medical, Synonym).
    <br>2. <b>Multi-Seed Mining:</b> Recursive scraping of Google Autocomplete for all variations.
    <br>3. <b>Back-Translation:</b> Uses AI to translate findings back to English.
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
    with st.spinner("Consulting German Lexicon (AI)..."):
        strategy = get_cultural_translation(api_key, keyword)
    
    if strategy:
        synonyms = strategy.get('synonyms', [])
        
        if not synonyms:
            st.error("AI returned empty synonyms. Try a different keyword.")
            st.stop()
        
        # 2. Deep Mine
        df_keywords = deep_mine_synonyms(synonyms)
        
        # 3. Back-Translate
        if not df_keywords.empty:
            with st.spinner(f"Translating {len(df_keywords)} keywords back to English..."):
                germ_list = df_keywords['German Keyword'].tolist()
                translations = batch_translate_to_english(api_key, germ_list)
                df_keywords['English Meaning'] = df_keywords['German Keyword'].map(translations).fillna("-")
        
        # 4. Trends
        main_term = synonyms[0]
        trend_data = get_google_trends([main_term])

        # --- OUTPUT ---
        st.markdown("---")
        st.markdown(f"### 1. Strategic Translation")
        st.info(f"**AI Insight:** {strategy.get('explanation', 'No context provided.')}")
        
        cols = st.columns(len(synonyms))
        for i, syn in enumerate(synonyms):
            cols[i].markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{syn}</div>
                <div class="metric-lbl">Variation #{i+1}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Trends
        st.markdown("---")
        st.subheader("2. Demand Trend")
        if trend_data is not None:
            fig = px.line(trend_data, y=main_term, title=f"Interest in '{main_term}' (Germany)", color_discrete_sequence=['#1a7f37'])
            fig.update_layout(plot_bgcolor='white', yaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Trend data unavailable (Google Trends API limits or low volume).")

        # Matrix
        st.markdown("---")
        st.subheader(f"3. The Master Matrix ({len(df_keywords)} Keywords)")
        
        if not df_keywords.empty:
            # Filter
            all_intents = ["All"] + list(df_keywords['Intent'].unique())
            sel_intent = st.selectbox("Filter by Intent:", all_intents)
            
            if sel_intent != "All":
                df_display = df_keywords[df_keywords['Intent'] == sel_intent]
            else:
                df_display = df_keywords
                
            # Reorder
            df_display = df_display[['German Keyword', 'English Meaning', 'Intent', 'Seed Term', 'Modifier Used']]
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "German Keyword": st.column_config.TextColumn("🇩🇪 German Query", width="medium"),
                    "English Meaning": st.column_config.TextColumn("🇺🇸 English Meaning", width="medium"),
                    "Seed Term": st.column_config.TextColumn("Source", width="small"),
                }
            )
            
            csv = df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Master List (CSV)", csv, "german_strategy.csv", "text/csv")
        else:
            st.warning("No keywords found. Google Autocomplete returned 0 results for these terms.")
        
    else:
        # This error now only shows if ALL models failed
        st.error("AI Analysis Failed. Please check your API Key quota.")

elif run_btn and not api_key:
    st.error("Enter API Key and Keyword.")
