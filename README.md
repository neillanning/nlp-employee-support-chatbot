# NLP Employee Support Chatbot

## Project Description
This is a **closed-domain**, voice-enabled retrieval-augmented generation (RAG) chatbot designed for internal company/employee support. It answers questions based solely on the documents provided in the `knowledge/` folder (Employee Handbook, How-To Guide, Company Directory, etc.).

The chatbot was developed as part of the NLP course final project. It demonstrates key NLP concepts including semantic retrieval, contrastive fine-tuning, and parameter-efficient adaptation (LoRA).

## Features
- Voice input (microphone button) using faster-whisper
- Text input with chat history
- Responses grounded in your uploaded company documents
- Fine-tuned embedder for better domain understanding
- LoRA-adapted generation for natural, colleague-like responses
- Fully local execution (no data leaves your machine when using the .exe)

## How to Run

### Option 1: Standalone Executable (Recommended - Best Performance)
1. Download the provided `.exe` file.
2. Double-click the executable to launch.
3. The chatbot will open in your browser at `http://localhost:8501`.
4. All company documents are already included.

### Option 2: Run from Source
1. Ensure Python 3.10+ and the virtual environment are set up.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

   Project Structureknowledge/ → Place all company PDFs, TXT, and MD files here
fine_tuned_models/ → Contains the fine-tuned embedder and LoRA adapters
chroma_db/ → Local vector database (created automatically)

Technical DetailsEmbedder: Fine-tuned all-MiniLM-L6-v2 using contrastive learning on Customer Support on Twitter, Ubuntu Dialogue Corpus v2.0, and RSiCS datasets.
Generator: Llama-3.2-3B-Instruct with LoRA adapters.
Speech-to-Text: faster-whisper
Vector Store: ChromaDB
UI Framework: Streamlit

Training Datasets UsedCustomer Support on Twitter
Ubuntu Dialogue Corpus v2.0
Relational Strategies in Customer Service (RSiCS)

