import re
import gradio as gr
from extract import extract_text_from_pdf ,ask_ollama, chunk_text, embed_chunks, store_in_chroma, query_chroma, get_model

# Preload the embedding model at startup
print("Loading embedding model...")
get_model()
print("Model loaded successfully!")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0c0c0f;
    --surface:   #13131a;
    --surface-hover: #1a1a24;
    --border:    #1e1e2e;
    --accent:    #00e5a0;
    --accent-dim:#00e5a022;
    --text:      #e2e2f0;
    --muted:     #5a5a7a;
    --sidebar-bg: #12121a;
    --card-bg:   #18181f;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--text) !important;
}

footer { display: none !important; }

#header {
    padding: 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

#header h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    color: var(--accent) !important;
    margin: 0 !important;
}

/* PDF List at Top - Scrollable */
#pdf-list-container {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    max-height: 180px;
    overflow-y: auto;
}

#pdf-list-container::-webkit-scrollbar {
    width: 6px;
}

#pdf-list-container::-webkit-scrollbar-track {
    background: var(--surface);
    border-radius: 3px;
}

#pdf-list-container::-webkit-scrollbar-thumb {
    background: var(--muted);
    border-radius: 3px;
}

#pdf-list-container h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--accent) !important;
    margin: 0 0 0.75rem 0 !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

#pdf-list-container .pdf-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--surface);
    border-radius: 6px;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: var(--text);
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

#pdf-list-container .pdf-item:hover {
    border-color: var(--accent-dim);
    background: var(--surface-hover);
}

#pdf-list-container .pdf-item:last-child {
    margin-bottom: 0;
}

#pdf-list-container .pdf-icon {
    color: #ff6b6b;
    font-size: 1rem;
}

#pdf-list-container .empty-state {
    color: var(--muted);
    font-size: 0.8rem;
    text-align: center;
    padding: 1rem;
}

#status {
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    padding: 0.4rem 0.8rem !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    margin-bottom: 1rem !important;
}

#chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

#msg-box textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

#send-btn {
    background: var(--accent) !important;
    color: black !important;
    border-radius: 6px !important;
}

#add-file-btn {
    max-width: 60px;
}

/* Friendly Sidebar */
#file-sidebar {
    background: var(--sidebar-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    height: 100%;
}

#file-sidebar .sidebar-title {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    color: var(--text) !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

#file-sidebar .sidebar-desc {
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    margin-bottom: 1rem !important;
    line-height: 1.4;
}

#file-sidebar .upload-area {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
}

#file-sidebar .upload-area:hover {
    border-color: var(--accent);
    background: var(--accent-dim);
}

#file-sidebar .upload-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

#file-sidebar .upload-text {
    font-size: 0.75rem;
    color: var(--muted);
}

#file-sidebar .file-count {
    font-size: 0.7rem;
    color: var(--accent);
    background: var(--accent-dim);
    padding: 0.25rem 0.5rem;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.75rem;
}
"""

def clean_response(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def handle_upload(files, chat_history, file_names):
    if not files:
        return chat_history, None, "No document loaded.", "_No thinking yet._", file_names, '<div class="empty-state">No documents loaded yet. Upload PDFs below to get started!</div>', "0 files loaded"

    chat_history = chat_history or []
    file_names = file_names or []
    collection = None

    try:
        for file in files:
            filename = file.name.split("/")[-1]

            if filename in file_names:
                continue

            text = extract_text_from_pdf(file.name)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            collection = store_in_chroma(chunks, embeddings, filename)

            file_names.append(filename)

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        return chat_history, None, "Failed.", "_No thinking yet._", file_names, "_No files yet_", "0 files loaded"

    # Build HTML for PDF list
    if file_names:
        pdf_items_html = "".join([
            f'<div class="pdf-item"><span class="pdf-icon">📄</span>{f}</div>' 
            for f in file_names
        ])
    else:
        pdf_items_html = '<div class="empty-state">No documents loaded yet. Upload PDFs below to get started!</div>'

    file_count_html = f"{len(file_names)} file{'s' if len(file_names) != 1 else ''} loaded"

    chat_history.append({
        "role": "assistant",
        "content": f"✅ Loaded {len(file_names)} file(s). Ask anything."
    })

    return chat_history, collection, "● Files loaded", "_Ready._", file_names, pdf_items_html, file_count_html


def submit_message(message, chat_history, collection):
    if not message.strip():
        yield "", chat_history, "_No thinking output._"
        return

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    if collection is None:
        chat_history.append({"role": "assistant", "content": "⚠️ Upload a PDF first."})
        yield "", chat_history, "_No thinking output._"
        return

    # Add an empty assistant message that will be updated during streaming
    chat_history.append({"role": "assistant", "content": "..."})
    
    # Immediate yield to show user message and thinking indicator
    yield "", chat_history, "_Thinking..._"

    try:
        question_embedding = embed_chunks([message]) 
        relevant_chunks, source = query_chroma(collection, question_embedding)

        context = "\n\n".join(relevant_chunks)
        print("relevant chunks:", relevant_chunks)

        full_response = ""
        full_thinking = ""

        # Stream the response
        for chunk, thinking_chunk in ask_ollama(context, message):
            if chunk is not None:
                full_response += chunk
                # Update the last assistant message with the accumulated response
                chat_history[-1]["content"] = clean_response(full_response)
                yield "", chat_history, "_Thinking..._"
            
            if thinking_chunk:
                full_thinking += thinking_chunk

        # Final update with complete thinking
        thinking_display = clean_response(full_thinking) if full_thinking else "_No thinking output._"
        chat_history[-1]["content"] = clean_response(full_response)
        yield "", chat_history, thinking_display

    except Exception as e:
        chat_history[-1]["content"] = f"Error: {e}"
        yield "", chat_history, "_Error occurred._"


def main():
    with gr.Blocks(title="DocChat") as demo:

        collection_state = gr.State(None)
        file_names_state = gr.State([])

        with gr.Column(elem_id="header"):
            gr.HTML("<h1>DOCCHAT</h1>")

        status_box = gr.Markdown("No document loaded.", elem_id="status")

        # PDF List at Top
        with gr.Column(elem_id="pdf-list-container"):
            gr.HTML("""
            <h3>📄 Loaded Documents</h3>
            <div id="pdf-items">
                <div class="empty-state">No documents loaded yet. Upload PDFs below to get started!</div>
            </div>
            """)

        with gr.Row():

            # Friendly Sidebar
            with gr.Column(scale=1, elem_id="file-sidebar"):
                gr.HTML("""
                <div class="sidebar-title">📂 Your Documents</div>
                <div class="sidebar-desc">Upload PDF files to chat with them. I'll read and understand their content.</div>
                """)
                
                add_file_btn = gr.File(
                    label="",
                    file_count="multiple",
                    file_types=[".pdf"],
                    elem_id="add-file-btn"
                )
                
                file_count_display = gr.HTML('<div class="file-count" id="file-count">0 files loaded</div>')

            # Main chat
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": "Upload a PDF to start."}],
                    height=500,
                    show_label=False,
                    elem_id="chatbot"
                )

                with gr.Row():
                    msg_box = gr.Textbox(placeholder="Ask something...", show_label=False, elem_id="msg-box", scale=9)
                    send_btn = gr.Button("Send", elem_id="send-btn", scale=1)

                with gr.Accordion("🧠 Thinking", open=False):
                    thinking_box = gr.Markdown("_No thinking yet._")

        # References for dynamic updates
        pdf_items_container = gr.HTML(elem_id="pdf-items")
        
        # Auto upload handler
        add_file_btn.change(
            fn=handle_upload,
            inputs=[add_file_btn, chatbot, file_names_state],
            outputs=[chatbot, collection_state, status_box, thinking_box, file_names_state, pdf_items_container, file_count_display]
        )

        send_btn.click(
            fn=submit_message,
            inputs=[msg_box, chatbot, collection_state],
            outputs=[msg_box, chatbot, thinking_box],
            show_progress=False
        )

        msg_box.submit(
            fn=submit_message,
            inputs=[msg_box, chatbot, collection_state],
            outputs=[msg_box, chatbot, thinking_box],
            show_progress=False
        )

    demo.launch(css=CSS)


if __name__ == "__main__":
    main()