import streamlit as st
import pandas as pd
import requests
import json
import time
import google.generativeai as genai
from pytrends.request import TrendReq
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION (Dejan Style)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Cross-Border Keyword Bridge", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    /* --- FORCE LIGHT MODE --- */
    :root { --primary-color: #1a7f37; --background-color: #ffffff; --secondary-background-color: #f6f8fa; --text-color: #24292e; --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .stApp { background-color: #ffffff; color: #24292e; }
    
    h1, h2, h3, h4 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 600; color: #111; letter-spacing: -0.3px; }
    p, li, span, div { color: #24292e; }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 20px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 10px;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #1a7f37; margin-bottom: 5px; }
    .metric-lbl { font-size: 0.8rem; color: #586069; text-transform: uppercase; letter-spacing: 0.5px; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    section[data-testid="stSidebar"] * { color: #24292e !important; }
    
    /* Inputs */
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; border-radius: 6px; }
    .stTextInput input:focus { border-color: #1a7f37 !important; box-shadow: 0 0 0 1px #1a7f37 !important; }
    
    /* Tech Note */
    .tech-note { font-size: 0.85rem; color: #57606a; background-color: #f6f8fa; border-left: 3px solid #0969da; padding: 12px; border-radius: 0 4px 4px 0; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stDataFrame"] { border: 1px solid #e1e4e8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC ENGINE
# -----------------------------------------------------------------------------

def get_cultural_translation(api_key, keyword):
    """Uses Gemini to get the SEO-Native German translation."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Act as a Native German SEO Expert.
        English Keyword: "{keyword}"
        
        Task:
        1. Provide the most common 'Search Term' used in Germany for this concept (The "Head Term").
        2. Provide 3 related English keywords and their German equivalents.
        
        Return raw JSON only:
        {{
            "primary_german": "keyword",
            "related_english": [
                {{"en": "rel1", "de": "trans1"}},
                {{"en": "rel2", "de": "trans2"}},
                {{"en": "rel3", "de": "trans3"}}
            ]
        }}
        """
        resp = model.generate_content(prompt)
        clean_text = resp.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_text)
    except Exception as e:
        return None

def fetch_suggestions(query, lang='de', gl='de'):
    """Base function to hit Google Suggest."""
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl={lang}&gl={gl}"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            return response.json()[1]
    except:
        pass
    return []

def get_deep_mining_suggestions(base_keyword):
    """
    The 'Alphabet Soup' Method.
    Appends German modifiers to force Google to reveal low-volume / long-tail queries.
    """
    # Strategic German Modifiers
    modifiers = [
        "",          # Raw (High Volume)
        " für",      # for (Specific Use Case)
        " bei",      # with/during (Condition)
        " gegen",    # against (Remedy)
        " ohne",     # without (Exclusion)
        " wann",     # when (Question)
        " was",      # what (Definition)
        " hausmittel", # home remedy (German favorite)
        " test",     # review
        " kaufen"    # transactional
    ]
    
    all_results = []
    
    # Progress bar container
    progress_text = "Mining Google Suggestions..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, mod in enumerate(modifiers):
        # Update progress
        my_bar.progress((i + 1) / len(modifiers), text=f"Mining: '{base_keyword}{mod}...'")
        
        search_term = f"{base_keyword}{mod}"
        results = fetch_suggestions(search_term)
        
        for r in results:
            # Categorize intent based on modifier
            intent = "General / Head"
            if "für" in mod: intent = "Specific Use"
            elif "gegen" in mod: intent = "Problem Solving"
            elif "hausmittel" in mod: intent = "Informational (Remedy)"
            elif "kaufen" in mod: intent = "Transactional"
            elif "test" in mod: intent = "Commercial Investigation"
            elif "wann" in mod or "was" in mod: intent = "Informational (Q&A)"
            
            all_results.append({"Keyword": r, "Source Context": f"'{mod.strip()}' modifier", "Intent": intent})
        
        time.sleep(0.1) # Be nice to Google API
        
    my_bar.empty()
    
    # Deduplicate
    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.drop_duplicates(subset=['Keyword'])
    return df

def get_google_trends(keywords):
    """Fetches relative search volume (0-100)."""
    try:
        pytrends = TrendReq(hl='de-DE', tz=360)
        target = keywords[:1] 
        pytrends.build_payload(target, cat=0, timeframe='today 12-m', geo='DE', gprop='')
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
    st.markdown("### ⚙️ Intelligence Config")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("""<a href="https://aistudio.google.com/app/apikey" target="_blank" style="font-size:0.8rem;color:#0969da;">🔑 Get Free Key</a>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    **Methodology:**
    <div class="tech-note">
    <b>1. Cultural Translation:</b> We use LLMs to find the "Native" term (e.g. "Handy" vs "Mobiltelefon").
    <br><b>2. Recursive Mining:</b> We iterate through German prepositions (<i>für, gegen, bei</i>) to force Google Autocomplete to reveal deep long-tail queries.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------

st.title("Cross-Border Keyword Bridge 🇩🇪")
st.markdown("### German Long-Tail Intelligence & Strategy")

with st.expander("How this tool works (Technical)", expanded=False):
    st.markdown("""
    **The Problem:** Direct translation misses search intent. "Diaper Rash" translates to "Windelausschlag" (Medical), but Germans search for "Wunder Po" (Colloquial).
    
    **The Solution:**
    1.  **AI Context:** We identify the colloquial "Money Keyword."
    2.  **Deep Scraping:** We use the "Alphabet Soup" method on Google Germany to find low-volume, high-intent long tails.
    3.  **Trend Validation:** We check seasonality using Google Trends data.
    """)

st.write("")
keyword = st.text_input("Enter English Keyword", placeholder="e.g. diaper rash")
run_btn = st.button("Generate German Strategy", type="primary")

if run_btn and keyword and api_key:
    
    # --- PHASE 1: TRANSLATION ---
    with st.spinner("Consulting German SEO Database..."):
        ai_data = get_cultural_translation(api_key, keyword)
    
    if ai_data:
        primary_de = ai_data['primary_german']
        
        # --- PHASE 2: DEEP MINING ---
        df_suggestions = get_deep_mining_suggestions(primary_de)
        
        # --- PHASE 3: TRENDS ---
        with st.spinner("Fetching Demand Trends..."):
            trend_data = get_google_trends([primary_de])

        # --- DASHBOARD ---
        st.markdown("---")
        st.subheader("1. Strategic Translation")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-val" style="color:#24292e;">{keyword}</div><div class="metric-lbl">English Input</div></div>""", unsafe_allow_html=True)
        with c2:
             st.markdown(f"""<div class="metric-card" style="border-left: 5px solid #1a7f37;"><div class="metric-val">{primary_de}</div><div class="metric-lbl">German Head Term</div></div>""", unsafe_allow_html=True)
        with c3:
             count = len(df_suggestions) if not df_suggestions.empty else 0
             st.markdown(f"""<div class="metric-card"><div class="metric-val">{count}</div><div class="metric-lbl">Variations Found</div></div>""", unsafe_allow_html=True)

        # --- TREND CHART ---
        st.markdown("---")
        st.subheader(f"2. Demand Trend (Head Term)")
        st.markdown(f"Relative interest in **Germany** over the last 12 months.")
        
        if trend_data is not None:
            fig = px.line(trend_data, y=primary_de, title=f"Demand: '{primary_de}'",
                          color_discrete_sequence=['#1a7f37'])
            fig.update_layout(plot_bgcolor='white', yaxis=dict(gridcolor='#f0f0f0'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trend data not available for this specific term (Volume might be low/stable).")

        # --- LONG TAIL TABLE ---
        st.markdown("---")
        st.subheader("3. The Long-Tail Matrix")
        st.markdown("Real queries scraped from Google Germany. Use these for **H2s** and **FAQ** sections.")
        
        if not df_suggestions.empty:
            # Intent Filter
            intents = ["All"] + list(df_suggestions['Intent'].unique())
            selected_intent = st.selectbox("Filter by Intent:", intents)
            
            if selected_intent != "All":
                df_display = df_suggestions[df_suggestions['Intent'] == selected_intent]
            else:
                df_display = df_suggestions

            st.dataframe(
                df_display, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Keyword": st.column_config.TextColumn("German Keyword", width="large"),
                    "Source Context": st.column_config.TextColumn("Discovery Modifier"),
                }
            )
            
            csv = df_suggestions.to_csv(index=False).encode('utf-8')
            st.download_button("Download Keyword List (CSV)", csv, "german_keywords.csv", "text/csv")
        else:
            st.warning("No suggestions found. The keyword might be too niche even for Google Suggest.")

        # --- MAPPINGS ---
        st.markdown("---")
        st.subheader("4. English Reference Mappings")
        st.caption("Use these to explain the keywords to English-speaking stakeholders.")
        
        rel_data = []
        for r in ai_data['related_english']:
            rel_data.append({"English Concept": r['en'], "German Equivalent": r['de']})
        st.dataframe(pd.DataFrame(rel_data), use_container_width=True, hide_index=True)

    else:
        st.error("AI Translation Failed. Please check API Key.")

elif run_btn and not api_key:
    st.error("Please enter a Gemini API Key in the sidebar.")