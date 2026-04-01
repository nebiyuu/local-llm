# DocChat — Local RAG PDF Chatbot

DocChat is a fully local, private, offline PDF chatbot built with RAG (Retrieval Augmented Generation). Upload one or more PDFs and ask questions — only the relevant parts of the document are sent to the model, not the whole thing.

No API keys. No data leaves your machine.

## How it works

Instead of dumping the entire PDF into the prompt, DocChat uses a two-step RAG pipeline:

**Indexing** (on upload)
1. Extract text from PDF using PyMuPDF
2. Split text into overlapping chunks
3. Embed each chunk into a vector using `sentence-transformers` (runs locally on CPU)
4. Store chunks + vectors in ChromaDB

**Retrieval** (on each question)
1. Embed the user's question using the same model
2. Query ChromaDB for the 3 most semantically similar chunks
3. Send only those chunks to Ollama as context
4. Return the answer

This means DocChat works on large documents without overwhelming the model's context window.

## Features

- 100% local — no API keys, no internet required after setup
- Multi-document support — upload multiple PDFs and search across all of them
- RAG pipeline built from scratch — no LangChain, no n8n
- Source tracking — knows which chunk came from which file
- Model thinking display — see the model's reasoning process
- Clean, minimal dark UI

## Stack

| Component | Library |
|---|---|
| UI | Gradio |
| PDF extraction | PyMuPDF |
| Chunking | plain Python |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector storage | ChromaDB |
| LLM inference | Ollama (`qwen3.5:0.8b`) |

## Quickstart

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama and pull the model:**
   ```bash
   ollama serve
   ollama pull qwen3.5:0.8b
   ```

3. **Run the app:**
   ```bash
   python ui.py
   ```

4. Open `http://127.0.0.1:7860` in your browser, upload a PDF, and start chatting.

## Docker

```bash
docker build -t docchat .
docker run -p 7860:7860 --add-host=host.docker.internal:host-gateway docchat
```

Ollama must be running on your host machine.

## File Overview

| File | Purpose |
|---|---|
| `ui.py` | Gradio interface and app logic |
| `extract.py` | PDF extraction, chunking, embedding, ChromaDB, Ollama |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container setup |

## Notes

- Embedding runs on CPU — no GPU required
- Tested with `qwen3.5:0.8b` but any Ollama model works
- ChromaDB stores in-memory per session — reloading the app clears the index