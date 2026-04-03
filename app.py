import streamlit as st
import os
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Fastest reliable model for Streamlit Cloud
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN is not set in Streamlit secrets.")
    st.stop()

KNOWLEDGE_DIR = "./knowledge"
CHROMA_PATH = "./chroma_db"

TOP_K = 2
MAX_NEW_TOKENS = 100

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return embedder, pipe

embedder, generation_pipe = load_models()

client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(name="company_knowledge")

def index_documents():
    if collection.count() > 0:
        st.sidebar.success(f"✅ Knowledge base loaded ({collection.count()} chunks)")
        return

    documents, metadatas, ids = [], [], []
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
            except:
                continue
        if not text.strip():
            continue
        for i in range(0, len(text), 400):
            chunk = text[i:i + 500]
            documents.append(chunk)
            metadatas.append({"source": filename})
            ids.append(f"{filename}_{i}")

    if documents:
        embeddings = embedder.encode(documents, show_progress_bar=True, batch_size=32)
        collection.add(documents=documents, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)
        st.sidebar.success(f"✅ Indexed {len(documents)} chunks")

with st.spinner("Loading knowledge base..."):
    index_documents()

def rag_query(question):
    query_emb = embedder.encode(question)
    results = collection.query(query_embeddings=[query_emb.tolist()], n_results=TOP_K)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""You are a professional company support assistant. Answer ONLY the question asked in 1-2 short sentences. Never ask follow-up questions or add extra text.

Company information:
{context}

Question: {question}

Answer:"""

    output = generation_pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.1)
    answer = output[0]['generated_text'].split("Answer:")[-1].strip() if "Answer:" in output[0]['generated_text'] else output[0]['generated_text'].replace(prompt, "").strip()
    return answer

st.title("🏢 Company Employee Support Chatbot")
st.caption("Closed-domain RAG • Fast & Stable")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about HR, IT, policies, or procedures...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        answer = rag_query(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

st.sidebar.info("Closed-domain RAG chatbot for employee support.")
