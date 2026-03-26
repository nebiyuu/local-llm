import re
import gradio as gr
from extract import extract_text_from_pdf, ask_ollama, chunk_text, embed_chunks, store_in_chroma, query_chroma

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0c0c0f;
    --surface:   #13131a;
    --border:    #1e1e2e;
    --accent:    #00e5a0;
    --accent-dim:#00e5a022;
    --text:      #e2e2f0;
    --muted:     #5a5a7a;
}

body, .gradio-container { background: var(--bg) !important; font-family: 'DM Mono', monospace !important; color: var(--text) !important; }
footer { display: none !important; }
#header { padding: 2rem 0 1.5rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
#header h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; font-size: 2rem !important; color: var(--accent) !important; letter-spacing: -0.03em; margin: 0 !important; }
#header p { color: var(--muted) !important; font-size: 0.8rem !important; margin: 0.3rem 0 0 0 !important; }
#status { font-size: 0.75rem !important; color: var(--muted) !important; padding: 0.4rem 0.8rem !important; background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; margin-bottom: 1rem !important; }
.upload-area { background: var(--surface) !important; border: 1px dashed var(--border) !important; border-radius: 8px !important; }
.upload-area:hover { border-color: var(--accent) !important; }
#load-btn { background: transparent !important; border: 1px solid var(--accent) !important; color: var(--accent) !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; border-radius: 4px !important; }
#load-btn:hover { background: var(--accent-dim) !important; }
#chatbot { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
#msg-box textarea { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: 'DM Mono', monospace !important; font-size: 0.85rem !important; border-radius: 6px !important; caret-color: var(--accent) !important; }
#msg-box textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-dim) !important; }
#send-btn { background: var(--accent) !important; border: none !important; color: #0c0c0f !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.85rem !important; border-radius: 6px !important; min-width: 80px !important; }
#send-btn:hover { opacity: 0.85 !important; }
#thinking-panel { background: var(--surface) !important; border: 1px solid var(--border) !important; border-left: 2px solid var(--muted) !important; border-radius: 8px !important; }
#thinking-content { color: var(--muted) !important; font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important; line-height: 1.7 !important; }
.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }
"""

def clean_response(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()

def handle_upload(files, chat_history):
    if not files:
        chat_history = chat_history or []
        chat_history.append({"role": "assistant", "content": "⚠️ No file selected. Please upload a PDF."})
        return chat_history, None, "No document loaded.", "_No thinking yet._"
    file = files[0]
    try:
        for file in files:
            text = extract_text_from_pdf(file.name)
            chunks = chunk_text(text)
            print(f"Total chunks: {len(chunks)}")
            embeddings = embed_chunks(chunks)
            print(f"Embedded {len(embeddings)} chunks")
            filename = file.name.split('/')[-1]
            collection = store_in_chroma(chunks, embeddings, filename)
            print("Stored in ChromaDB")
    except Exception as e:
        chat_history = chat_history or []
        chat_history.append({"role": "assistant", "content": f"❌ Could not read PDF: {e}"})
        return chat_history, None, "Failed to load document.", "_No thinking yet._"
    word_count = len(text.split())
    chat_history = chat_history or []
    chat_history.append({"role": "assistant", "content": f"✅ Document loaded — {word_count:,} words · {len(chunks)} chunks. Ask me anything about it."})
    status = f"● {file.name.split('/')[-1]}  ·  {word_count:,} words  ·  {len(chunks)} chunks"
    return chat_history, collection, status, "_No thinking yet — ask a question first._"

def submit_message(message, chat_history, collection):
    if not message.strip():
        return "", chat_history, gr.update()
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    if collection is None:
        chat_history.append({"role": "assistant", "content": "⚠️ No document loaded. Please upload a PDF first."})
        return "", chat_history, gr.update()

    try:
        # embed the question and retrieve relevant chunks
        question_embedding = embed_chunks([message])
        relevant_chunks,source = query_chroma(collection, question_embedding)
        context = "\n\n".join(relevant_chunks)
        print(f"Retrieved {len(relevant_chunks)} chunks for query")

        raw_answer, raw_thinking = ask_ollama(context, message)
        answer   = clean_response(raw_answer   or "(no response)")
        thinking = clean_response(raw_thinking or "")
    except Exception as e:
        answer   = f"❌ Error: {e}"
        thinking = ""

    chat_history.append({"role": "assistant", "content": answer})
    thinking_display = thinking if thinking else "_Thinking was not enabled or produced no output for this response._"
    return "", chat_history, thinking_display

def main():
    with gr.Blocks(title="DocChat") as demo:
        # now stores a chroma collection object instead of raw text
        collection_state = gr.State(None)

        with gr.Column(elem_id="header"):
            gr.HTML("<h1>DOCCHAT</h1><p>local · private · offline — powered by qwen via ollama</p>")

        status_box = gr.Markdown("No document loaded.", elem_id="status")

        with gr.Row():
            pdf_input = gr.File(label="Upload PDF", file_count="multiple", file_types=[".pdf"], elem_classes=["upload-area"])
            load_btn = gr.Button("LOAD →", elem_id="load-btn", scale=0)

        gr.HTML("<hr class='divider'>")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Upload a PDF above to get started."}],
                    elem_id="chatbot",
                    height=460,
                    show_label=False,
                )
                with gr.Row(elem_id="input-row"):
                    msg_box = gr.Textbox(placeholder="Ask a question about your document...", show_label=False, elem_id="msg-box", scale=9)
                    send_btn = gr.Button("SEND", elem_id="send-btn", scale=1)

            with gr.Column(scale=2):
                with gr.Accordion("🧠  Model Thinking", open=False, elem_id="thinking-panel"):
                    thinking_box = gr.Markdown(value="_No thinking yet — ask a question first._", elem_id="thinking-content")

        load_btn.click(fn=handle_upload, inputs=[pdf_input, chatbot], outputs=[chatbot, collection_state, status_box, thinking_box])
        send_btn.click(fn=submit_message, inputs=[msg_box, chatbot, collection_state], outputs=[msg_box, chatbot, thinking_box])
        msg_box.submit(fn=submit_message, inputs=[msg_box, chatbot, collection_state], outputs=[msg_box, chatbot, thinking_box])

    demo.launch(css=CSS)

if __name__ == "__main__":
    main()