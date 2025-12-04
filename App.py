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
# 1. VISUAL CONFIGURATION (Dejan Style - Light Mode Forced)
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
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; color: #24292e !important; }
    
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stDataFrame"] { border: 1px solid #e1e4e8; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOGIC ENGINE (Dynamic Model Selector)
# -----------------------------------------------------------------------------

def get_cultural_translation(api_key, keyword):
    """
    Uses Gemini to get the SEO-Native German translation.
    Loops through available models until one succeeds.
    """
    genai.configure(api_key=api_key)
    
    # Priority List: Fastest/Cheapest -> Most Capable -> Legacy
    candidates = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "gemini-pro"
    ]
    
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
    
    last_error = None
    
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            
            # Clean response
            text = resp.text
            if "```" in text:
                text = re.sub(r"```json|```", "", text).strip()
            
            return json.loads(text)
            
        except Exception as e:
            last_error = e
            continue # Try next model
            
    st.error(f"All AI models failed. Last error: {str(last_error)}")
    return None

def batch_translate_to_english(api_key, german_keywords):
    """
    Translates German keywords back to English using Dynamic Model Selection.
    """
    if not german_keywords: return {}
    
    genai.configure(api_key=api_key)
    candidates = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    # Chunking to avoid token limits
    targets = german_keywords[:50] 
    
    prompt = f"""
    Translate these German search queries into English. Keep it short and literal.
    Input List: {json.dumps(targets)}
    
    Return ONLY raw JSON key-value pairs:
    {{
        "German Keyword": "English Translation"
    }}
    """
    
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            
            text = resp.text
            if "```" in text:
                text = re.sub(r"```json|```", "", text).strip()
            
            return json.loads(text)
        except:
            continue
            
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
    """
    Recursive Mining with Progress Bar.
    """
    modifiers = ["", " für", " bei", " gegen", " was", " wann", " hausmittel", " kosten", " tipps", " kaufen"]
    all_data = []
    
    progress_bar = st.progress(0, text="Initializing Mining...")
    total_steps = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
