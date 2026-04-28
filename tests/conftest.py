import os
import pytest
import tempfile
import fitz


@pytest.fixture
def sample_text():
    """Sample text for chunking tests."""
    return (
        "This is the first sentence of the document. "
        "Here is the second sentence with more content. "
        "The third sentence continues the narrative. "
        "Fourth sentence adds more details. "
        "Fifth sentence brings additional information. "
        "Sixth sentence expands further. "
        "Seventh sentence provides more context. "
        "Eighth sentence is the last one in this sample."
    )


@pytest.fixture
def sample_text_long():
    """Longer sample text for comprehensive chunking tests."""
    words = []
    for i in range(500):
        words.append(f"word{i}")
    return " ".join(words)


@pytest.fixture
def sample_chunks():
    """Pre-defined chunks for embedding tests."""
    return [
        "The first chunk contains important information about the topic.",
        "The second chunk continues with additional details and facts.",
        "The third chunk provides supplementary content for context.",
    ]


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a temporary PDF file for testing."""
    pdf_path = tmp_path / "test.pdf"

    doc = fitz.open()
    for page_num in range(3):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"This is page {page_num + 1}.\n\n"
            f"Contains some sample text for testing.\n"
            f"Line number {page_num * 3 + 3}.\n"
            f"Another line here."
        )
    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


@pytest.fixture
def sample_pdf_single_page(tmp_path):
    """Create a single-page PDF for simpler tests."""
    pdf_path = tmp_path / "single_page.pdf"

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Single page test content.\n"
        "Another line of text.\n"
        "Final line here."
    )
    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


@pytest.fixture
def chroma_client():
    """Create a fresh ChromaDB client for each test."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(Settings(
        persist_directory=None,
        anonymized_telemetry=False
    ))

    yield client

    try:
        client.delete_collection("test_collection")
    except Exception:
        pass


@pytest.fixture
def chroma_collection(chroma_client):
    """Create a temporary collection for testing."""
    collection = chroma_client.get_or_create_collection(name="test_collection")
    yield collection
    try:
        chroma_client.delete_collection("test_collection")
    except Exception:
        pass


@pytest.fixture
def mock_ollama_stream_response():
    """Mock streaming response from Ollama."""
    import json

    lines = [
        json.dumps({"response": "Hello", "done": False}),
        json.dumps({"response": " there", "done": False}),
        json.dumps({"response": "!", "done": True}),
    ]
    return lines


@pytest.fixture
def mock_ollama_thinking_response():
    """Mock streaming response with thinking from Ollama."""
    import json

    lines = [
        json.dumps({"response": "The answer is", "done": False, "thinking": "Let me think..."}),
        json.dumps({"response": " 42", "done": False, "thinking": ""}),
        json.dumps({"response": ".", "done": True, "thinking": "Found it!"}),
    ]
    return lines


@pytest.fixture
def mock_ollama_error_response():
    """Mock error response from Ollama."""
    return b'{"error": "Connection refused"}'


@pytest.fixture
def temp_chroma_persist_dir(tmp_path):
    """Temporary directory for ChromaDB persistence testing."""
    persist_dir = tmp_path / "chroma_db"
    persist_dir.mkdir()
    return str(persist_dir)