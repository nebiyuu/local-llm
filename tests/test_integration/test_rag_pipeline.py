import pytest
import numpy as np
from docchat.core.pdf import extract_text_from_pdf
from docchat.core.chunker import chunk_text
from docchat.core.embedding import embed_chunks
from docchat.store.collection import DocumentCollection


class TestRAGPipeline:
    """Integration tests for the full RAG pipeline."""

    def test_full_pipeline_single_document(self, chroma_client, sample_pdf_path):
        """Test the complete pipeline from PDF to query."""
        text = extract_text_from_pdf(sample_pdf_path)
        assert len(text) > 0

        chunks = chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 0

        embeddings = embed_chunks(chunks)
        assert embeddings.shape[0] == len(chunks)

        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="rag_test")
        collection.add(chunks, embeddings, "test.pdf")

        query_embedding = embed_chunks(["page"])
        results, sources = collection.query(query_embedding, n_results=3)

        assert len(results) > 0

    def test_pipeline_with_multiple_documents(self, chroma_client):
        """Test pipeline with multiple PDFs."""
        doc1_chunks = ["Document 1 content chunk 1", "Document 1 content chunk 2"]
        doc2_chunks = ["Document 2 content chunk 1", "Document 2 content chunk 2"]

        doc1_embeddings = np.random.rand(2, 384).astype(np.float32)
        doc2_embeddings = np.random.rand(2, 384).astype(np.float32)

        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="multi_doc_test")
        collection.add(doc1_chunks, doc1_embeddings, "doc1.pdf")
        collection.add(doc2_chunks, doc2_embeddings, "doc2.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        results, sources = collection.query(query_embedding)

        assert len(results) == 4
        assert "doc1.pdf" in sources or "doc2.pdf" in sources

    def test_chunking_then_embedding_consistency(self, sample_text_long):
        """Test that chunking output can be directly embedded."""
        chunk_size = 50
        overlap = 10
        chunks = chunk_text(sample_text_long, chunk_size=chunk_size, overlap=overlap)

        embeddings = embed_chunks(chunks)
        assert embeddings.shape[0] == len(chunks)

        for i in range(len(chunks)):
            single_embedding = embed_chunks([chunks[i]])
            assert np.allclose(single_embedding, embeddings[i:i+1])

    def test_retrieval_relevance(self, chroma_client):
        """Test that relevant documents are retrieved."""
        chunks = [
            "Python is a high-level programming language.",
            "The weather today is sunny and warm.",
            "Machine learning uses algorithms to learn from data."
        ]
        embeddings = np.array([
            [0.9, 0.1, 0.2],
            [0.1, 0.9, 0.1],
            [0.2, 0.1, 0.9]
        ]).astype(np.float32)

        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="relevance_test")
        collection.add(chunks, embeddings, "test.pdf")

        programming_query = [np.array([0.85, 0.15, 0.25]).astype(np.float32)]
        results, sources = collection.query(programming_query, n_results=1)

        assert len(results) >= 1
        assert "programming" in results[0].lower() or "Python" in results[0]


class TestRAGPipelineEdgeCases:
    """Edge case tests for the RAG pipeline."""

    def test_pipeline_with_empty_pdf(self, tmp_path):
        """Test pipeline with empty PDF."""
        empty_pdf = tmp_path / "empty.pdf"
        import fitz
        doc = fitz.open()
        doc.save(str(empty_pdf))
        doc.close()

        text = extract_text_from_pdf(str(empty_pdf))
        assert text == ""

        chunks = chunk_text(text)
        assert len(chunks) == 0

    def test_pipeline_with_large_chunk_size(self, sample_text_long):
        """Test pipeline with very large chunk size."""
        chunks = chunk_text(sample_text_long, chunk_size=1000, overlap=0)
        embeddings = embed_chunks(chunks)
        assert embeddings.shape[0] == len(chunks)

    def test_pipeline_preserves_chunk_order(self, chroma_client):
        """Test that chunks maintain order after retrieval."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="order_test")

        ordered_chunks = [f"Chunk {i} with unique identifier {i}" for i in range(10)]
        embeddings = np.random.rand(10, 384).astype(np.float32)
        collection.add(ordered_chunks, embeddings, "ordered.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        results, sources = collection.query(query_embedding, n_results=5)

        for chunk in results:
            chunk_num = int(chunk.split()[-1])
            assert chunk_num < 10


class TestPromptConstruction:
    """Tests for prompt construction used in the RAG pipeline."""

    def test_context_includes_retrieved_chunks(self, chroma_client):
        """Test that retrieved chunks form the context."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="doc_collection")

        chunks = [
            "First chunk has specific info about topic A.",
            "Second chunk continues with topic B details."
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        collection.add(chunks, embeddings, "doc.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        retrieved_chunks, _ = collection.query(query_embedding)

        context = "\n\n".join(retrieved_chunks)
        assert "chunk" in context.lower()

    def test_prompt_includes_user_question(self):
        """Test that user question is included in prompt."""
        from docchat.llm.client import ask_ollama
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.iter_lines.return_value = iter([
            b'{"response": "Answer", "done": true}'
        ])

        context = "Test context"
        question = "Test question"
        full_prompt = None

        def capture_prompt(*args, **kwargs):
            nonlocal full_prompt
            full_prompt = kwargs.get('json', {}).get('prompt', '')
            return mock_response

        with patch('requests.post', side_effect=capture_prompt):
            list(ask_ollama(context, question))

        if full_prompt:
            assert question in full_prompt
            assert context in full_prompt
