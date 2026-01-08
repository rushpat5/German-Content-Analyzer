import json
import random
import time
import logging
from typing import Optional, List, Dict

import pandas as pd
import requests
import streamlit as st
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# CONFIG & STYLE
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Everyday User Prompts (Multi-Lang)", layout="wide", page_icon="🌍")

st.markdown(
    """
    <style>
        :root { --brand: #3182ce; --bg: #ffffff; }
        .stApp { background-color: var(--bg); font-family: sans-serif; }
        
        .user-bubble {
            background: #eebbbb; /* Light Red/Pink for User */
            padding: 12px 18px;
            border-radius: 18px 18px 18px 0px;
            margin-top: 5px;
            color: #2c3e50;
            font-size: 1.05rem;
            font-weight: 500;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }
        
        .translation-text {
            color: #718096;
            font-size: 0.9rem;
            font-style: italic;
            margin-top: 6px;
            margin-left: 5px;
            display: flex;
            align-items: center;
        }

        .intent-label {
            font-size: 0.75rem;
            color: #a0aec0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 2px;
        }

        .container-box {
            margin-bottom: 25px;
            border-bottom: 1px solid #edf2f7;
            padding-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LANGUAGE CONFIGURATION
# ---------------------------------------------------------
# This dictionary maps the dropdown selection to search params and modifiers
LANG_CONFIG = {
    "German (Germany)": {
        "hl": "de", "gl": "de", "llm_lang": "German",
        "modifiers": [
            " schreib mir", " erklär mir", " hilf mir", " ideen für", 
            " beispiele für", " wie mache ich", " tipps gegen", " unterschied zwischen"
        ]
    },
    "French (France)": {
        "hl": "fr", "gl": "fr", "llm_lang": "French",
        "modifiers": [
            " écris moi", " explique moi", " aide moi", " idées pour", 
            " exemples de", " comment faire", " conseils pour", " différence entre"
        ]
    },
    "Spanish (Spain)": {
        "hl": "es", "gl": "es", "llm_lang": "Spanish",
        "modifiers": [
            " escribe un", " explícame", " ayúdame", " ideas para", 
            " ejemplos de", " cómo hacer", " consejos para", " diferencia entre"
        ]
    },
    "English (UK)": {
        "hl": "en", "gl": "gb", "llm_lang": "British English",
        "modifiers": [
            " write me", " explain to me", " help me", " ideas for", 
            " examples of", " how to do", " tips for", " difference between"
        ]
    },
    "English (US)": {
        "hl": "en", "gl": "us", "llm_lang": "American English",
        "modifiers": [
            " write me", " explain to me", " help me", " ideas for", 
            " examples of", " how to do", " tips for", " difference between"
        ]
    }
}

# ---------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(hf_token: Optional[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if hf_token:
            return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except:
        pass
    return SentenceTransformer("all-MiniLM-L6-v2").to(device)

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

# ---------------------------------------------------------
# GROQ WRAPPER
# ---------------------------------------------------------
def run_groq(api_key: str, prompt: str) -> Dict:
    client = get_groq_client(api_key)
    models = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]
    
    for m in models:
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=m,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            continue
    return {"error": "API Error"}

# ---------------------------------------------------------
# 1. TRANSLATE & CONTEXTUALIZE TOPIC
# ---------------------------------------------------------
def get_localized_context(api_key: str, topic: str, target_lang: str) -> List[str]:
    """Get simple terms in the target language to seed the search."""
    prompt = f"""
    Translate the topic "{topic}" into 3 common {target_lang} search terms used by everyday people.
    Return JSON: {{ "terms": ["term1", "term2", "term3"] }}
    """
    res = run_groq(api_key, prompt)
    return res.get("terms", []) if "error" not in res else []

# ---------------------------------------------------------
# 2. MINE "CHAT-STYLE" INTENTS
# ---------------------------------------------------------
def fetch_suggestions(q: str, hl: str, gl: str) -> List[str]:
    # Dynamic Google URL based on host language (hl) and geo-location (gl)
    url = f"https://www.google.com/complete/search?client=chrome&q={q}&hl={hl}&gl={gl}"
    try:
        time.sleep(random.uniform(0.1, 0.2))
        r = requests.get(url, timeout=2.0)
        # Handle potential empty responses
        if r.status_code == 200 and len(r.json()) > 1:
            return [x for x in r.json()[1] if isinstance(x, str)]
        return []
    except:
        return []

def mine_conversational_intents(seeds: List[str], modifiers: List[str], hl: str, gl: str) -> pd.DataFrame:
    rows = []
    prog = st.progress(0, "Mining everyday questions...")
    
    total_steps = len(seeds)
    for i, s in enumerate(seeds):
        prog.progress((i + 1) / total_steps)
        for m in modifiers:
            query = f"{s}{m}" 
            results = fetch_suggestions(query, hl, gl)
            for r in results:
                rows.append({"Raw Search": r, "Topic": s})
                
    prog.empty()
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["Raw Search"])

# ---------------------------------------------------------
# 3. SIMULATE NATURAL USER PROMPTS
# ---------------------------------------------------------
def simulate_user_prompts(api_key: str, searches: List[str], target_lang: str) -> Dict[str, Dict[str, str]]:
    """
    Takes a search query and returns:
    1. Natural Prompt in Target Language
    2. English Translation (if target is English, it returns the same or a clean version)
    """
    if not searches: return {}
    
    unique_searches = list(set(searches))
    results = {}
    chunk_size = 8
    
    prog = st.progress(0, f"Simulating user typing in {target_lang}...")
    
    for i in range(0, len(unique_searches), chunk_size):
        chunk = unique_searches[i:i + chunk_size]
        prog.progress(i / len(unique_searches))
        
        prompt = f"""
        You are simulating an average {target_lang} user typing into an AI Assistant.
        
        Task: 
        1. Convert the Search Query into a NATURAL, CASUAL {target_lang} Prompt.
        2. Provide an English translation of that prompt (if the prompt is already English, provide the same text).
        
        Input Example (if target was German):
        "bewerbung schreiben hilfe" -> {{
            "natural_prompt": "Kannst du mir helfen, eine Bewerbung zu schreiben? Ich weiß nicht, wie ich anfangen soll.",
            "english": "Can you help me write a job application? I don't know how to start."
        }}

        Convert these queries: {json.dumps(chunk, ensure_ascii=False)}

        Return JSON: {{ "mapping": {{ "query_key": {{ "natural_prompt": "...", "english": "..." }} }} }}
        """
        
        res = run_groq(api_key, prompt)
        if "error" not in res:
            mapping = res.get("mapping", {})
            results.update(mapping)
            
        time.sleep(0.5)
        
    prog.empty()
    return results

# ---------------------------------------------------------
# UI MAIN
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    # Language Selector
    selected_lang_name = st.selectbox(
        "Select Target Language", 
        options=list(LANG_CONFIG.keys()),
        index=0 # Defaults to German
    )
    
    # Retrieve config based on selection
    current_config = LANG_CONFIG[selected_lang_name]
    
    api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("HF Token (Optional)", type="password")
    num_results = st.slider("Number of Prompts", 5, 30, 10)

st.title(f"🗣️ Everyday User Prompts ({selected_lang_name})")
st.markdown(f"Discover how **real people** in **{current_config['llm_lang']}** ask AI for help (with English translations).")

topic = st.text_input(f"Enter Topic (e.g., Cooking, Office) - You can type in English or {current_config['llm_lang']}")

if st.button("Generate User Prompts"):
    if not api_key or not topic:
        st.error("Need API Key and Topic.")
        st.stop()
        
    st.session_state.df = None
    
    # 1. Get Context (Localized)
    seeds = get_localized_context(api_key, topic, current_config['llm_lang'])
    if not seeds:
        seeds = [topic]
        
    st.write(f"🔎 *Searching using localized terms: {', '.join(seeds)}*")

    # 2. Mine Data (with localized modifiers and Google params)
    df_raw = mine_conversational_intents(
        seeds, 
        current_config['modifiers'], 
        current_config['hl'], 
        current_config['gl']
    )
    
    if df_raw.empty:
        st.warning("No data found. Try a different topic.")
        st.stop()
        
    # 3. Filter Relevance
    model = load_embedding_model(hf_token)
    embeddings = model.encode(df_raw["Raw Search"].tolist())
    topic_emb = model.encode(topic) # Embed the user input topic
    df_raw["Score"] = util.cos_sim(embeddings, topic_emb).cpu().numpy()
    
    # Take top N
    df_top = df_raw.sort_values("Score", ascending=False).head(num_results)
    
    # 4. Humanize & Translate
    human_map = simulate_user_prompts(
        api_key, 
        df_top["Raw Search"].tolist(), 
        current_config['llm_lang']
    )
    
    # Extract columns
    df_top["Natural Prompt"] = df_top["Raw Search"].map(lambda x: human_map.get(x, {}).get("natural_prompt", x))
    df_top["English Translation"] = df_top["Raw Search"].map(lambda x: human_map.get(x, {}).get("english", "-"))
    
    # Remove failed generations
    df_top = df_top[df_top["English Translation"] != "-"]

    # Display Results
    st.markdown(f"### 💬 What {current_config['llm_lang']} users are typing:")
    st.markdown("---")
    
    for _, row in df_top.iterrows():
        prompt_text = row['Natural Prompt']
        english = row['English Translation']
        raw_intent = row['Raw Search']
        
        # UI Logic: If language is English, maybe don't emphasize translation as much,
        # but for consistency we keep the design.
        
        st.markdown(f"""
        <div class="container-box">
            <div class='intent-label'>Original Search Intent: {raw_intent}</div>
            <div class='user-bubble'>
                User: <b>"{prompt_text}"</b>
            </div>
            <div class='translation-text'>
                🇬🇧 "{english}"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # CSV Download
    csv_data = df_top[["Raw Search", "Natural Prompt", "English Translation"]].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV (Bilingual)", csv_data, f"user_prompts_{current_config['hl']}.csv", "text/csv")
