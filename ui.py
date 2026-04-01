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

#file-sidebar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem;
}
"""

def clean_response(text: str) -> str:
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


def handle_upload(files, chat_history, file_names):
    if not files:
        return chat_history, None, "No document loaded.", "_No thinking yet._", file_names, "_No files yet_"

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
        return chat_history, None, "Failed.", "_No thinking yet._", file_names, "\n".join(file_names)

    file_display = "\n".join([f"- {f}" for f in file_names])

    chat_history.append({
        "role": "assistant",
        "content": f"✅ Loaded {len(file_names)} file(s). Ask anything."
    })

    return chat_history, collection, "● Files loaded", "_Ready._", file_names, file_display


def submit_message(message, chat_history, collection):
    if not message.strip():
        return "", chat_history, gr.update()

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    if collection is None:
        chat_history.append({"role": "assistant", "content": "⚠️ Upload a PDF first."})
        return "", chat_history, gr.update()

    try:
        question_embedding = embed_chunks([message])
        relevant_chunks, source = query_chroma(collection, question_embedding)
        context = "\n\n".join(relevant_chunks)

        raw_answer, raw_thinking = ask_ollama(context, message)

        answer = clean_response(raw_answer or "(no response)")
        thinking = clean_response(raw_thinking or "")

    except Exception as e:
        answer = f"❌ Error: {e}"
        thinking = ""

    chat_history.append({"role": "assistant", "content": answer})

    thinking_display = thinking if thinking else "_No thinking output._"

    return "", chat_history, thinking_display


def main():
    with gr.Blocks(title="DocChat") as demo:

        collection_state = gr.State(None)
        file_names_state = gr.State([])

        with gr.Column(elem_id="header"):
            gr.HTML("<h1>DOCCHAT</h1>")

        status_box = gr.Markdown("No document loaded.", elem_id="status")

        with gr.Row():

            # Sidebar
            with gr.Column(scale=1, elem_id="file-sidebar"):
                gr.Markdown("### 📂 Docs")
                file_list = gr.Markdown("_No files yet_")

                add_file_btn = gr.File(
                    label="➕",
                    file_count="multiple",
                    file_types=[".pdf"],
                    elem_id="add-file-btn"
                )

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

        # AUTO upload (no button)
        add_file_btn.change(
            fn=handle_upload,
            inputs=[add_file_btn, chatbot, file_names_state],
            outputs=[chatbot, collection_state, status_box, thinking_box, file_names_state, file_list]
        )

        send_btn.click(
            fn=submit_message,
            inputs=[msg_box, chatbot, collection_state],
            outputs=[msg_box, chatbot, thinking_box]
        )

        msg_box.submit(
            fn=submit_message,
            inputs=[msg_box, chatbot, collection_state],
            outputs=[msg_box, chatbot, thinking_box]
        )

    demo.launch(css=CSS)


if __name__ == "__main__":
    main()