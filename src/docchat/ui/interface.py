import re
import gradio as gr
from ..core.pdf import extract_text_from_pdf
from ..core.chunker import chunk_text
from ..core.embedding import embed_chunks, get_model
from ..store.collection import DocumentCollection
from ..llm.client import ask_ollama
from .styles import CSS


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
            collection = DocumentCollection()
            collection.add(chunks, embeddings.tolist(), filename)

            file_names.append(filename)

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        return chat_history, None, "Failed.", "_No thinking yet._", file_names, "_No files yet_", "0 files loaded"

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

    chat_history.append({"role": "assistant", "content": "..."})
    yield "", chat_history, "_Thinking..._"

    try:
        question_embedding = embed_chunks([message])
        relevant_chunks, source = collection.query(question_embedding.tolist())

        context = "\n\n".join(relevant_chunks)
        print("relevant chunks:", relevant_chunks)

        full_response = ""
        full_thinking = ""

        for chunk, thinking_chunk in ask_ollama(context, message):
            if chunk is not None:
                full_response += chunk
                chat_history[-1]["content"] = clean_response(full_response)
                yield "", chat_history, "_Thinking..._"
            
            if thinking_chunk:
                full_thinking += thinking_chunk

        thinking_display = clean_response(full_thinking) if full_thinking else "_No thinking output._"
        chat_history[-1]["content"] = clean_response(full_response)
        yield "", chat_history, thinking_display

    except Exception as e:
        chat_history[-1]["content"] = f"Error: {e}"
        yield "", chat_history, "_Error occurred._"


def main():
    print("Loading embedding model...")
    get_model()
    print("Model loaded successfully!")

    with gr.Blocks(title="DocChat") as demo:
        collection_state = gr.State(None)
        file_names_state = gr.State([])

        with gr.Column(elem_id="header"):
            gr.HTML("<h1>DOCCHAT</h1>")

        status_box = gr.Markdown("No document loaded.", elem_id="status")

        with gr.Column(elem_id="pdf-list-container"):
            gr.HTML("""
            <h3>📄 Loaded Documents</h3>
            <div id="pdf-items">
                <div class="empty-state">No documents loaded yet. Upload PDFs below to get started!</div>
            </div>
            """)

        with gr.Row():
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

        pdf_items_container = gr.HTML(elem_id="pdf-items")
        
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