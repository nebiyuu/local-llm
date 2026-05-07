import pytest
import numpy as np
from docchat.store.collection import DocumentCollection


class TestDocumentCollection:
    """Tests for the DocumentCollection class."""

    def test_add_single_chunk(self, chroma_client, sample_chunks):
        """Test adding a single chunk."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        embeddings = np.random.rand(1, 384).astype(np.float32)
        result = collection.add([sample_chunks[0]], embeddings, "test_doc.pdf")
        assert result is not None

    def test_add_multiple_chunks(self, chroma_client, sample_chunks):
        """Test adding multiple chunks."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        collection.add(sample_chunks, embeddings, "test_doc.pdf")

        result = collection.collection.get()
        assert len(result["documents"]) == len(sample_chunks)

    def test_add_with_unique_ids(self, chroma_client, sample_chunks):
        """Test that added documents have unique IDs."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        embeddings = np.random.rand(2, 384).astype(np.float32)
        collection.add(sample_chunks[:2], embeddings, "test_doc.pdf")

        result = collection.collection.get()
        ids = result["ids"]
        assert len(ids) == len(set(ids))

    def test_add_preserves_metadata(self, chroma_client):
        """Test that metadata is correctly stored."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        chunks = ["Test chunk content"]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        collection.add(chunks, embeddings, "example.pdf")

        result = collection.collection.get()
        sources = result["metadatas"]
        assert all(m["source"] == "example.pdf" for m in sources)

    def test_multiple_docs_same_collection(self, chroma_client):
        """Test adding documents from multiple files in same collection."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        chunks1 = ["Content from doc 1"]
        chunks2 = ["Content from doc 2"]
        embeddings = np.random.rand(2, 384).astype(np.float32)

        collection.add(chunks1, embeddings[:1], "doc1.pdf")
        collection.add(chunks2, embeddings[1:], "doc2.pdf")

        result = collection.collection.get()
        assert len(result["documents"]) == 2


class TestQueryCollection:
    """Tests for the query method."""

    def test_query_returns_results(self, chroma_client, sample_chunks):
        """Test that query returns relevant results."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        collection.add(sample_chunks, embeddings, "test.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        chunks, sources = collection.query(query_embedding)

        assert len(chunks) > 0
        assert len(sources) > 0

    def test_query_respects_n_results(self, chroma_client, sample_chunks):
        """Test that n_results parameter is respected."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        collection.add(sample_chunks, embeddings, "test.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        chunks, sources = collection.query(query_embedding, n_results=2)

        assert len(chunks) <= 2

    def test_query_returns_sources(self, chroma_client):
        """Test that sources are correctly returned."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        chunks = ["Test content"]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        collection.add(chunks, embeddings, "source.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        chunks, sources = collection.query(query_embedding)

        assert all(s == "source.pdf" for s in sources)

    def test_query_empty_collection(self, chroma_client):
        """Test querying an empty collection."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        chunks, sources = collection.query(query_embedding)

        assert len(chunks) == 0
        assert len(sources) == 0

    def test_reset_collection(self, chroma_client, sample_chunks):
        """Test resetting the collection."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        collection.add(sample_chunks, embeddings, "test.pdf")

        collection.reset()
        result = collection.collection.get()
        assert len(result["documents"]) == 0


class TestChromaIntegration:
    """Integration tests for ChromaDB store and query."""

    def test_store_and_query_roundtrip(self, chroma_client):
        """Test storing and then querying a document."""
        collection = DocumentCollection()
        collection.collection = chroma_client.get_or_create_collection(name="test_collection")

        chunks = [
            "Python is a programming language.",
            "JavaScript is used for web development.",
            "Machine learning is a subset of AI."
        ]
        embeddings = np.random.rand(len(chunks), 384).astype(np.float32)
        collection.add(chunks, embeddings, "programming.pdf")

        query_embedding = [np.random.rand(384).astype(np.float32)]
        results, sources = collection.query(query_embedding)

        assert len(results) > 0
        assert len(results) == len(sources)
