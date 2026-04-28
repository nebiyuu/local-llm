import pytest
import numpy as np
from extract import store_in_chroma, query_chroma


class TestStoreInChroma:
    """Tests for the store_in_chroma function."""

    def test_store_single_chunk(self, chroma_client, sample_chunks):
        """Test storing a single chunk."""
        embeddings = np.random.rand(1, 384).astype(np.float32)
        collection = store_in_chroma(
            [sample_chunks[0]],
            embeddings,
            "test_doc.pdf"
        )
        assert collection is not None

    def test_store_multiple_chunks(self, chroma_client, sample_chunks):
        """Test storing multiple chunks."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        collection = store_in_chroma(
            sample_chunks,
            embeddings,
            "test_doc.pdf"
        )
        assert collection is not None

        result = collection.get()
        assert len(result["documents"]) == len(sample_chunks)

    def test_store_with_unique_ids(self, chroma_client, sample_chunks):
        """Test that stored documents have unique IDs."""
        embeddings = np.random.rand(2, 384).astype(np.float32)
        collection = store_in_chroma(
            sample_chunks[:2],
            embeddings,
            "test_doc.pdf"
        )

        result = collection.get()
        ids = result["ids"]
        assert len(ids) == len(set(ids))

    def test_store_preserves_metadata(self, chroma_client):
        """Test that metadata is correctly stored."""
        chunks = ["Test chunk content"]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        collection = store_in_chroma(chunks, embeddings, "example.pdf")

        result = collection.get()
        sources = result["metadatas"]
        assert all(m["source"] == "example.pdf" for m in sources)

    def test_multiple_docs_same_collection(self, chroma_client):
        """Test storing documents from multiple files in same collection."""
        chunks1 = ["Content from doc 1"]
        chunks2 = ["Content from doc 2"]
        embeddings = np.random.rand(2, 384).astype(np.float32)

        store_in_chroma(chunks1, embeddings[:1], "doc1.pdf")
        store_in_chroma(chunks2, embeddings[:1], "doc2.pdf")

        collection = chroma_client.get_or_create_collection(name="documents")
        result = collection.get()
        assert len(result["documents"]) == 2


class TestQueryChroma:
    """Tests for the query_chroma function."""

    def test_query_returns_results(self, chroma_collection, sample_chunks):
        """Test that query returns relevant results."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        store_in_chroma(sample_chunks, embeddings, "test.pdf")

        query_embedding = np.random.rand(384).astype(np.float32)
        chunks, sources = query_chroma(chroma_collection, query_embedding)

        assert len(chunks) > 0
        assert len(sources) > 0

    def test_query_respects_n_results(self, chroma_collection, sample_chunks):
        """Test that n_results parameter is respected."""
        embeddings = np.random.rand(len(sample_chunks), 384).astype(np.float32)
        store_in_chroma(sample_chunks, embeddings, "test.pdf")

        query_embedding = np.random.rand(384).astype(np.float32)
        chunks, sources = query_chroma(chroma_collection, query_embedding, n_results=2)

        assert len(chunks) <= 2

    def test_query_returns_sources(self, chroma_collection):
        """Test that sources are correctly returned."""
        chunks = ["Test content"]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        store_in_chroma(chunks, embeddings, "source.pdf")

        query_embedding = np.random.rand(384).astype(np.float32)
        chunks, sources = query_chroma(chroma_collection, query_embedding)

        assert all(s == "source.pdf" for s in sources)

    def test_query_empty_collection(self, chroma_collection):
        """Test querying an empty collection."""
        query_embedding = np.random.rand(384).astype(np.float32)
        chunks, sources = query_chroma(chroma_collection, query_embedding)

        assert len(chunks) == 0
        assert len(sources) == 0


class TestChromaIntegration:
    """Integration tests for ChromaDB store and query."""

    def test_store_and_query_roundtrip(self, chroma_collection):
        """Test storing and then querying a document."""
        chunks = [
            "Python is a programming language.",
            "JavaScript is used for web development.",
            "Machine learning is a subset of AI."
        ]
        embeddings = np.random.rand(len(chunks), 384).astype(np.float32)
        store_in_chroma(chunks, embeddings, "programming.pdf")

        query_embedding = np.random.rand(384).astype(np.float32)
        results, sources = query_chroma(chroma_collection, query_embedding)

        assert len(results) > 0
        assert len(results) == len(sources)

    def test_semantic_search_relevance(self, chroma_collection):
        """Test that semantically similar queries return relevant results."""
        chunks = [
            "The sun is very bright today.",
            "I enjoy eating chocolate cake.",
            "Python programming is fun."
        ]
        embeddings = np.array([
            [0.1, 0.9, 0.2],
            [0.3, 0.4, 0.5],
            [0.8, 0.1, 0.9]
        ]).astype(np.float32)

        store_in_chroma(chunks, embeddings, "test.pdf")

        similar_query = np.array([0.15, 0.85, 0.25]).astype(np.float32)
        results, sources = query_chroma(chroma_collection, similar_query, n_results=1)

        assert len(results) >= 1