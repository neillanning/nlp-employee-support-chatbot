import streamlit as st
import os
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import pypdf
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN is not set in Streamlit secrets.")
    st.stop()

EMBED_MODEL_PATH = "./fine_tuned_models/fine_tuned_embedder"
KNOWLEDGE_DIR = "./knowledge"
CHROMA_PATH = "./chroma_db"

TOP_K = 2
MAX_NEW_TOKENS = 70

@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_PATH, device="cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return embedder, pipe, tokenizer

embedder, generation_pipe, tokenizer = load_models()

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

    prompt = f"""You are a company support assistant. Answer the question directly and stop.

Rules:
- Answer in 1 short sentence only.
- Use only the company information below.
- Never generate any new question.
- Never output "Question:", "Follow-up", "Would you like", or similar.
- Stop completely after the answer.

Company information:
{context}

Question: {question}

Answer:"""

    output = generation_pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.1,
        top_p=0.7,
        repetition_penalty=1.8,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )

    full_text = output[0]['generated_text']

    if "Answer:" in full_text:
        answer = full_text.split("Answer:")[-1].strip()
    else:
        answer = full_text.replace(prompt, "").strip()

    # Aggressive hard cutoff
    stop_phrases = ["Question:", "Follow-up", "Would you like", "Do you have", "Let me know", 
                    "Anything else", "Next question", "Another question"]
    for phrase in stop_phrases:
        if phrase in answer:
            answer = answer.split(phrase)[0].strip()
            break

    # Take only the first complete sentence
    sentences = answer.split('. ')
    if len(sentences) > 1:
        answer = sentences[0] + '.'

    return answer

st.title("🏢 Company Employee Support Chatbot")
st.caption("Closed-domain RAG • Strictly one answer only (Llama-3.2-3B)")

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

st.sidebar.info("""**NLP Chatbot Project**
- Closed-domain RAG using Llama-3.2-3B-Instruct
- Retrieval with fine-tuned Sentence Transformers + ChromaDB
- Responses grounded strictly in your company documents""")
