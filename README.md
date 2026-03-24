# DocChat: Local PDF Chatbot

DocChat is a local, private, offline chatbot that lets you upload a PDF and ask questions about its content. Powered by [Gradio](https://gradio.app/) for the UI, [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF extraction, and [Ollama](https://ollama.com/) for local LLM inference (using the Qwen model).

## Features
- Upload a PDF and chat with it instantly
- All processing is local and private
- Modern, beautiful Gradio UI
- Model thinking/trace display

## Quickstart

1. **Install requirements** (preferably in a virtual environment):
	```bash
	pip install -r requirments.txt
	```

2. **Start Ollama** and pull the Qwen model (if not already running):
	```bash
	ollama serve
	ollama pull qwen3.5:0.8b(or a model of you choice)
	```

3. **Run the app:**
	```bash
	python ui.py
	```

4. **Open the Gradio link** in your browser at *local URL:  http://127.0.0.1:7865*, upload a PDF, and start chatting!

## File Overview

- `ui.py` — Gradio chat interface
- `extract.py` — PDF extraction and model query logic
- `requirments.txt` — Python dependencies

## Notes
- Requires [Ollama](https://ollama.com/) running locally on port 11434
- Only PDF files are supported for upload
- All data stays on your machine



