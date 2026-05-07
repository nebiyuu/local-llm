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