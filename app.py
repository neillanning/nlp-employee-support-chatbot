import streamlit as st
import os
import io
import wave
import numpy as np
import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sounddevice as sd
import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# ---------------- Configuration ----------------
WHISPER_MODEL_SIZE = "medium"
EMBED_MODEL_PATH = "./fine_tuned_models/fine_tuned_embedder"
LORA_ADAPTER_PATH = "./fine_tuned_models/lora_adapters"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Secure token loading
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN environment variable is not set. Please set it before running the app.")
    st.stop()

CHROMA_PATH = "./chroma_db"
KNOWLEDGE_DIR = "knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3  # Reduced for faster performance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE,
                                 compute_type="float16" if DEVICE == "cuda" else "int8")
    embedder = SentenceTransformer(EMBED_MODEL_PATH, device=DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model = model.to(DEVICE)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1
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
            except Exception as e:
                st.sidebar.warning(f"Could not read PDF {filename}: {e}")
                continue

        else:
            continue

        if not text.strip():
            continue

        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk_start": i})
            ids.append(f"{filename}_{i}")

    if not documents:
        st.error("No valid documents found in 'knowledge/' folder!")
        return

    embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=32)
    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )
    st.sidebar.success(f"Indexed {len(documents)} chunks from {len(set(m['source'] for m in metadatas))} files")


with st.spinner("Checking / building knowledge index..."):
    index_documents()


# ---------------- Audio Recording ----------------
def record_audio(duration=10, fs=16000):
    st.info("Recording… Speak now (10 seconds)")
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


# ---------------- Speech-to-Text ----------------
def transcribe_audio(audio_bytes):
    segments, _ = whisper_model.transcribe(io.BytesIO(audio_bytes), beam_size=5, language="en")
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


# ---------------- RAG Query ----------------
def rag_query(question, history=""):
    query_emb = embedder.encode(question)
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=TOP_K)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a friendly, knowledgeable company colleague helping employees.
Be concise, helpful, natural. Use contractions (you're, it's). Avoid sounding robotic.

Previous conversation:
{history}

Relevant company information:
{context}

Question: {question}

Answer:"""

    output = generation_pipe(
        prompt,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        max_length=None,
        num_return_sequences=1
    )
    return output[0]['generated_text'].strip().replace(prompt, "").strip()


# ---------------- Streamlit UI ----------------
st.title("Company Employee Support Chatbot")
st.caption("Voice or text questions about policies, IT, procedures — uses your trained models & knowledge files.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.chat_input("Type your question…")

with col2:
    if st.button("🎤 Speak (10 sec)", use_container_width=True):
        audio_data = record_audio()
        with st.spinner("Transcribing…"):
            query = transcribe_audio(audio_data)
        if query.strip():
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Thinking…"):
                history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
                answer = rag_query(query, history_str)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking…"):
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]])
        answer = rag_query(prompt, history_str)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Sidebar
st.sidebar.header("Chatbot Info")
st.sidebar.markdown("""
**Final Trained Version**  
- Voice → faster-whisper  
- Retrieval → Fine-tuned embedder  
- Generation → Llama-3.2-3B with LoRA adapters  
- All local — no cloud  

**Knowledge files**  
Place your PDFs, TXT, MD files in the `knowledge/` folder next to this script.
""")