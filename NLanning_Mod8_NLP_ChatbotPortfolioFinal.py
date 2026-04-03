import sys
import os
import io
import wave
import numpy as np
import torch
import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sounddevice as sd
import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# ---------------- Critical Fixes ----------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------- Robust Path Configuration ----------------
if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

WHISPER_MODEL_SIZE = "medium"
EMBED_MODEL_PATH = os.path.join(BASE_PATH, "fine_tuned_models", "fine_tuned_embedder")
LORA_ADAPTER_PATH = os.path.join(BASE_PATH, "fine_tuned_models", "lora_adapters")
KNOWLEDGE_DIR = os.path.join(BASE_PATH, "knowledge")
CHROMA_PATH = os.path.join(BASE_PATH, "chroma_db")

BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN environment variable is not set. Please set it before running.")
    st.stop()

TOP_K = 3
MAX_NEW_TOKENS = 120

print(f"Base path: {BASE_PATH}")

# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    print(f"Loading models on device: {device}")

    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    embedder = SentenceTransformer(EMBED_MODEL_PATH, device=device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )

    if LORA_ADAPTER_PATH and os.path.exists(LORA_ADAPTER_PATH):
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        print("LoRA adapters loaded successfully")
    else:
        model = base_model
        print("Using base model (no LoRA)")

    model = model.to(device)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id
    )

    return whisper_model, embedder, pipe, tokenizer


whisper_model, embedder, generation_pipe, tokenizer = load_models()

# ---------------- Chroma Vector Store ----------------
client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(name="company_knowledge")

# ---------------- Index Knowledge Base ----------------
def index_documents():
    if collection.count() > 0:
        st.sidebar.success(f"Using existing index with {collection.count()} chunks")
        return

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(KNOWLEDGE_DIR):
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        text = ""

        if filename.lower().endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif filename.lower().endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(filepath)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                st.sidebar.warning(f"Could not read PDF {filename}")
                continue

        if not text.strip():
            continue

        for i in range(0, len(text), 400):
            chunk = text[i:i + 500]
            documents.append(chunk)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_{i}")

    if documents:
        embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=16)
        collection.add(documents=documents, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)
        st.sidebar.success(f"Indexed {len(documents)} chunks")

with st.spinner("Loading knowledge base..."):
    index_documents()

# ---------------- Audio Functions ----------------
def record_audio(duration=10, fs=16000):
    st.info("Recording... Speak now (10 seconds)")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    st.success("Recording finished.")
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(fs)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
    return byte_io.getvalue()

def transcribe_audio(audio_bytes):
    segments, _ = whisper_model.transcribe(io.BytesIO(audio_bytes), beam_size=5, language="en")
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())

# ---------------- RAG Query (No follow-up questions) ----------------
def rag_query(question, history=""):
    query_emb = embedder.encode(question)
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=TOP_K)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a helpful, professional company colleague.
Answer the question clearly and concisely based ONLY on the provided company information.
Do NOT add any follow-up questions, suggestions, or extra commentary.
Use natural language and contractions.

Previous conversation:
{history}

Relevant company information:
{context}

Question: {question}

Answer:"""

    output = generation_pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        max_length=None,
        num_return_sequences=1
    )

    answer = output[0]['generated_text'].replace(prompt, "").strip()

    # Safety: cut off any follow-up if the model still adds it
    if "Follow-up" in answer or "What happens if" in answer:
        answer = answer.split("Follow-up")[0].split("What happens if")[0].strip()

    return answer

# ---------------- Streamlit UI ----------------
st.title("Company Employee Support Chatbot")
st.caption("Voice-enabled • Trained on your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.button("🎤 Speak (10 sec)"):
    audio_data = record_audio()
    with st.spinner("Transcribing..."):
        query = transcribe_audio(audio_data)
    if query.strip():
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Thinking..."):
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
            answer = rag_query(query, history_str)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

prompt = st.chat_input("Type your question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
        answer = rag_query(prompt, history_str)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

st.sidebar.info("Closed-domain RAG chatbot with voice input.\nAll processing is local.")
