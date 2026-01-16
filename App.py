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

st.set_page_config(page_title="Everyday User Prompts (DE)", layout="wide", page_icon="🗣️")

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
# CACHED RESOURCES
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(hf_token: Optional[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Using a standard sentence transformer for filtering
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)
    except:
        return SentenceTransformer("all-MiniLM-L6-v2").to("cpu")

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

# ---------------------------------------------------------
# HUGGING FACE INFERENCE (TranslateGemma)
# ---------------------------------------------------------
def translate_with_gemma(text: str, hf_token: str) -> str:
    """
    Uses Google's TranslateGemma via HF Inference API.
    """
    if not hf_token:
        return "Error: No HF Token provided"

    # API Endpoint for TranslateGemma
    # Note: If this specific model endpoint is busy/gated, you can swap the URL 
    # to "https://api-inference.huggingface.co/models/google/gemma-2-9b-it"
    API_URL = "https://api-inference.huggingface.co/models/google/translategemma-4b-it"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # TranslateGemma specific prompt structure or standard instruction
    payload = {
        "inputs": f"Translate the following text from English to German: {text}",
        "parameters": {"max_new_tokens": 128, "return_full_text": False}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Handle "Model Loading" state (HF serverless cold boot)
        if response.status_code == 503:
            time.sleep(2) # Wait a bit and retry once
            response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            # HF returns list of dicts: [{'generated_text': '...'}]
            if isinstance(result, list) and 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
            return str(result)
        else:
            return f"Translation Error ({response.status_code})"
    except Exception as e:
        return f"API Error: {str(e)}"

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
def get_german_context(api_key: str, topic: str) -> List[str]:
    prompt = f"""
    Translate the topic "{topic}" into 3 common German search terms used by everyday people.
    Return JSON: {{ "terms": ["term1", "term2", "term3"] }}
    """
    res = run_groq(api_key, prompt)
    return res.get("terms", []) if "error" not in res else []

# ---------------------------------------------------------
# 2. MINE "CHAT-STYLE" INTENTS
# ---------------------------------------------------------
def fetch_suggestions(q: str) -> List[str]:
    url = f"https://www.google.com/complete/search?client=chrome&q={q}&hl=de&gl=de"
    try:
        time.sleep(random.uniform(0.1, 0.2))
        r = requests.get(url, timeout=2.0)
        return [x for x in r.json()[1] if isinstance(x, str)]
    except:
        return []

def mine_conversational_intents(seeds: List[str]) -> pd.DataFrame:
    modifiers = [
        " schreib mir", " erklär mir", " hilf mir",
        " ideen für", " beispiele für", 
        " was ist der unterschied", " zusammenfassung",
        " wie mache ich", " tipps gegen"
    ]
    
    rows = []
    prog = st.progress(0, "Mining everyday questions...")
    
    for i, s in enumerate(seeds):
        for m in modifiers:
            prog.progress((i + 1) / len(seeds))
            query = f"{s}{m}" 
            results = fetch_suggestions(query)
            for r in results:
                rows.append({"Raw Search": r, "Topic": s})
                
    prog.empty()
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["Raw Search"])

# ---------------------------------------------------------
# 3. GENERATE (GROQ) & TRANSLATE (HF GEMMA)
# ---------------------------------------------------------
def simulate_user_prompts(api_key: str, hf_token: str, searches: List[str]) -> Dict[str, Dict[str, str]]:
    """
    1. Groq generates Natural English Prompt.
    2. HF Gemma translates English Prompt -> German.
    """
    if not searches: return {}
    
    unique_searches = list(set(searches))
    results = {}
    chunk_size = 5 # Small chunks
    
    prog = st.progress(0, "Generating prompts (Llama) & Translating (Gemma)...")
    
    for i in range(0, len(unique_searches), chunk_size):
        chunk = unique_searches[i:i + chunk_size]
        prog.progress(i / len(unique_searches))
        
        # 1. GENERATE ENGLISH PROMPT WITH GROQ
        prompt_groq = f"""
        You are analyzing German Search Queries.
        Task: Create a natural, casual **English** User Prompt for an AI Chatbot based on the intent of the German search.
        
        Input: "bewerbung schreiben hilfe"
        Output Structure:
        "bewerbung schreiben hilfe": {{
            "english": "Can you help me write a job application? I don't know how to start."
        }}

        Convert these queries: {json.dumps(chunk, ensure_ascii=False)}
        Return JSON: {{ "mapping": {{ "query_key": {{ "english": "..." }} }} }}
        """
        
        res_groq = run_groq(api_key, prompt_groq)
        
        # 2. TRANSLATE WITH HF GEMMA
        if "error" not in res_groq:
            mapping = res_groq.get("mapping", {})
            
            for key, data in mapping.items():
                english_text = data.get("english", "")
                
                if english_text:
                    # Call HF API
                    german_translation = translate_with_gemma(english_text, hf_token)
                    
                    results[key] = {
                        "english": english_text,
                        "german": german_translation
                    }
        
        time.sleep(0.5) # Respect Rate limits
        
    prog.empty()
    return results

# ---------------------------------------------------------
# UI MAIN
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Required for Google Gemma Translation")
    num_results = st.slider("Number of Prompts", 5, 30, 10)

st.title("🗣️ Everyday User Prompts (DE)")
st.markdown("Mines German search intents -> **Llama 3** generates English prompts -> **Google TranslateGemma** translates to German.")

topic = st.text_input("Enter Topic (e.g., Cooking, Office, Dating)")
if st.button("Generate User Prompts"):
    if not api_key or not topic or not hf_token:
        st.error("Please provide Groq API Key, Topic, and Hugging Face Token.")
        st.stop()
        
    st.session_state.df = None
    
    # 1. Get German Context
    seeds = get_german_context(api_key, topic)
    if not seeds: seeds = [topic]
        
    # 2. Mine Data
    df_raw = mine_conversational_intents(seeds)
    if df_raw.empty:
        st.warning("No data found. Try a different topic.")
        st.stop()
        
    # 3. Filter Relevance
    model = load_embedding_model(hf_token) # Token irrelevant for MiniLM but kept signature
    embeddings = model.encode(df_raw["Raw Search"].tolist())
    topic_emb = model.encode(topic)
    df_raw["Score"] = util.cos_sim(embeddings, topic_emb).cpu().numpy()
    
    # Take top N
    df_top = df_raw.sort_values("Score", ascending=False).head(num_results)
    
    # 4. Generate & Translate
    human_map = simulate_user_prompts(api_key, hf_token, df_top["Raw Search"].tolist())
    
    # Extract columns
    df_top["English Prompt"] = df_top["Raw Search"].map(lambda x: human_map.get(x, {}).get("english", "-"))
    df_top["German Prompt (Gemma)"] = df_top["Raw Search"].map(lambda x: human_map.get(x, {}).get("german", "-"))
    
    # Remove failed generations
    df_top = df_top[df_top["English Prompt"] != "-"]

    # Display Results
    st.markdown("### 🇩🇪 Generated & Translated Prompts:")
    st.markdown("---")
    
    for _, row in df_top.iterrows():
        german = row['German Prompt (Gemma)']
        english = row['English Prompt']
        raw_intent = row['Raw Search']
        
        st.markdown(f"""
        <div class="container-box">
            <div class='intent-label'>Original Search Intent: {raw_intent}</div>
            <div class='user-bubble'>
                User (English): <b>"{english}"</b>
            </div>
            <div class='translation-text'>
                🇩🇪 Gemma Translation: "{german}"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # CSV Download
    csv_data = df_top[["Raw Search", "English Prompt", "German Prompt (Gemma)"]].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv_data, "user_prompts_gemma.csv", "text/csv")
