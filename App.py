import streamlit as st
import pandas as pd
import requests
import json
import time
from groq import Groq
import re
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------------------------------------------------------------
# 1. VISUAL CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="German SEO Planner (Llama 3.3)", layout="wide", page_icon="🇩🇪")

st.markdown("""
<style>
    :root { --primary-color: #f55036; --background-color: #ffffff; --text-color: #24292e; }
    .stApp { background-color: #ffffff; color: #24292e; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    .metric-card { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 8px; padding: 15px; text-align: center; }
    .metric-val { font-size: 1.2rem; font-weight: 700; color: #f55036; margin-bottom: 5px; }
    section[data-testid="stSidebar"] { background-color: #f6f8fa; border-right: 1px solid #d0d7de; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d0d7de !important; }
    
    /* Status Badge */
    .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .status-ok { background-color: #dafbe1; color: #1a7f37; }
    .status-err { background-color: #ffebe9; color: #cf222e; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SESSION STATE
# -----------------------------------------------------------------------------
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_results = None
    st.session_state.synonyms = []
    st.session_state.strategy_text = ""
    st.session_state.working_groq_model = None # Memory for the working model

# -----------------------------------------------------------------------------
# 3. VECTOR ENGINE (Google Gemma via HuggingFace)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gemma_model(hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except Exception as e:
        return SentenceTransformer("all-MiniLM-L6-v2")

def process_keywords_gemma(df_keywords, seeds, threshold, hf_token):
    model = load_gemma_model(hf_token)
    if not model: return None
    
    try:
        seed_vecs = model.encode(seeds, prompt_name="STS", normalize_embeddings=True)
        candidates = df_keywords['German Keyword'].tolist()
        candidate_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    except TypeError:
        seed_vecs = model.encode(seeds, normalize_embeddings=True)
        candidates = df_keywords['German Keyword'].tolist()
        candidate_vecs = model.encode(candidates, normalize_embeddings=True)
    
    scores_matrix = util.cos_sim(candidate_vecs, seed_vecs)
    max_scores, _ = torch.max(scores_matrix, dim=1)
    
    df_keywords['Relevance'] = max_scores.numpy()
    return df_keywords[df_keywords['Relevance'] >= threshold].sort_values('Relevance', ascending=False)

# -----------------------------------------------------------------------------
# 4. GENERATIVE ENGINE (GROQ - SELF HEALING)
# -----------------------------------------------------------------------------
def run_groq(api_key, prompt):
    client = Groq(api_key=api_key)
    
    # PRIORITY LIST (Newest to Oldest)
    # If one is decommissioned, it automatically tries the next one.
    candidates = [
        "llama-3.3-70b-versatile",   # Best current model (Dec 2024/Jan 2025)
        "llama-3.1-70b-versatile",   # Previous Standard
        "llama-3.2-90b-vision-preview", # Strong alternative
        "llama-3.1-8b-instant"       # Fastest / Backup
    ]
    
    # Use cached model if we found one already
    if st.session_state.working_groq_model:
        candidates.insert(0, st.session_state.working_groq_model)
    
    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))

    last_error = None

    for model_name in candidates:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful SEO assistant. Output strict JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # If it worked, save this model as the "Working" one
            st.session_state.working_groq_model = model_name
            return json.loads(chat_completion.choices[0].message.content)

        except Exception as e:
            last_error = str(e)
            
            # If Decommissioned (400) or Not Found (404), continue to next model
            if "404" in last_error or "400" in last_error or "decommissioned" in last_error:
                continue
            
            # If Rate Limit (429), wait and retry ONCE on the same model
            if "429" in last_error:
                time.sleep(2)
                try:
                    chat_completion = client.chat.completions.create(
                         messages=[{"role": "user", "content": prompt}],
                         model=model_name, temperature=0.1, response_format={"type": "json_object"}
                    )
                    st.session_state.working_groq_model = model_name
                    return json.loads(chat_completion.choices[0].message.content)
                except:
                    continue # Move to next model if retry fails

            # Invalid Key
            if "401" in last_error:
                return {"error": "INVALID_KEY"}
                
    return {"error": f"All Groq models failed. Last error: {last_error}"}

def get_cultural_translation(api_key, keyword):
    prompt = f"""
    Act as a Native German SEO Expert.
    English Keyword: "{keyword}"
    
    Task: Identify the top 3 distinct German terms used for this concept. 
    1. The most common colloquial term.
    2. The formal/medical term.
    3. A popular synonym.
    
    Output JSON format:
    {{ 
        "synonyms": ["term1", "term2", "term3"], 
        "explanation": "Brief reasoning in English" 
    }}
    """
    return run_groq(api_key, prompt)

def batch_translate(api_key, keywords):
    if not keywords: return {}
    # Groq handles larger contexts well
    chunks = [keywords[i:i+50] for i in range(0, len(keywords), 50)]
    full_map = {}
    
    for chunk in chunks:
        prompt = f"""
        Task: Translate these German keywords to English literally.
        Input List: {json.dumps(chunk)}
        
        Output JSON format:
        {{ "GermanKeyword": "EnglishTranslation", ... }}
        """
        res = run_groq(api_key, prompt)
        if "error" not in res: full_map.update(res)
        time.sleep(0.5) 
    return full_map

# -----------------------------------------------------------------------------
# 5. MINING & MAIN APP
# -----------------------------------------------------------------------------
def fetch_suggestions(query):
    url = f"http://google.com/complete/search?client=chrome&q={query}&hl=de&gl=de"
    try:
        r = requests.get(url, timeout=1.5)
        if r.status_code == 200: return r.json()[1]
    except: pass
    return []

def deep_mine(synonyms):
    modifiers = ["", " kaufen", " test", " vergleich", " kosten", " erfahrung", " beste"]
    all_data = []
    prog = st.progress(0, "Mining Google Autocomplete...")
    total = len(synonyms) * len(modifiers)
    step = 0
    
    for seed in synonyms:
        for mod in modifiers:
            step += 1
            if total > 0: prog.progress(min(step/total, 1.0))
            
            results = fetch_suggestions(f"{seed}{mod}")
            intent = "Informational"
            if "kaufen" in mod or "kosten" in mod: intent = "Transactional"
            elif "test" in mod or "vergleich" in mod or "beste" in mod: intent = "Commercial"
            elif "erfahrung" in mod: intent = "Review"
            
            for r in results:
                all_data.append({"German Keyword": r, "Seed": seed, "Intent": intent})
            time.sleep(0.05)
            
    prog.empty()
    df = pd.DataFrame(all_data)
    if not df.empty: return df.drop_duplicates(subset=['German Keyword'])
    return df

# --- UI ---
with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input("Groq API Key", type="password", help="Get free key at console.groq.com")
    hf_token = st.text_input("Hugging Face Token", type="password")
    st.markdown("---")
    threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.45, 0.05)
    
    if st.session_state.working_groq_model:
        st.success(f"Using: {st.session_state.working_groq_model}")

    # Check if Groq lib is installed
    try:
        import groq
        st.markdown('<span class="status-badge status-ok">✓ Groq Library Detected</span>', unsafe_allow_html=True)
    except ImportError:
        st.markdown('<span class="status-badge status-err">⚠ Groq Missing in requirements.txt</span>', unsafe_allow_html=True)

st.title("German SEO Planner 🇩🇪 (Llama 3.3)")
st.markdown("### High-Speed Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic", placeholder="e.g. coffee machines")
run_btn = st.button("Generate Keywords", type="primary")

if run_btn and keyword and api_key and hf_token:
    st.session_state.data_processed = False
    
    # 1. LOAD VECTORS
    with st.spinner("Loading Vector Model..."):
        try: _ = load_gemma_model(hf_token)
        except: st.stop()

    # 2. STRATEGY (Llama 3.3)
    with st.spinner("Analyzing Cultural Context (Llama 3.3)..."):
        strategy = get_cultural_translation(api_key, keyword)
        
        # ERROR HANDLING
        if not strategy: st.error("No response."); st.stop()
        if "error" in strategy:
            err = strategy['error']
            if err == "INVALID_KEY": st.error("❌ Invalid Groq API Key."); st.stop()
            st.error(f"Error: {err}")
            st.stop()
            
        st.session_state.synonyms = strategy.get('synonyms', [])
        st.session_state.strategy_text = strategy.get('explanation', '')

    # 3. MINE
    df = deep_mine(st.session_state.synonyms)
    
    if not df.empty:
        with st.spinner(f"Filtering {len(df)} keywords..."):
            df_filtered = process_keywords_gemma(df, st.session_state.synonyms, threshold, hf_token)
            
        if df_filtered is not None and not df_filtered.empty:
            with st.spinner("Translating..."):
                top = df_filtered.head(40)['German Keyword'].tolist()
                trans_map = batch_translate(api_key, top)
                df_filtered['English'] = df_filtered['German Keyword'].map(trans_map).fillna("-")
            
            st.session_state.df_results = df_filtered
            st.session_state.data_processed = True
        else: st.warning("No keywords met threshold.")
    else: st.warning("No keywords found.")

if st.session_state.data_processed:
    st.success(f"**Context:** {st.session_state.strategy_text}")
    cols = st.columns(len(st.session_state.synonyms))
    for i, syn in enumerate(st.session_state.synonyms):
        if i < len(cols): cols[i].markdown(f"""<div class="metric-card"><div class="metric-val">{syn}</div></div>""", unsafe_allow_html=True)
    
    df_show = st.session_state.df_results
    csv = df_show.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "keywords.csv", "text/csv")
    
    st.dataframe(df_show[['German Keyword', 'English', 'Intent', 'Relevance']], use_container_width=True, hide_index=True, column_config={"Relevance": st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=1)})
