import pytest
from extract import get_model, embed_chunks


class TestGetModel:
    """Tests for the get_model function."""

    def test_model_returns_singleton(self):
        """Test that get_model returns the same instance on multiple calls."""
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2

    def test_model_is_loaded(self):
        """Test that model is properly loaded and functional."""
        model = get_model()
        assert model is not None
        assert hasattr(model, 'encode')

    @pytest.mark.skipif(
        not hasattr(__import__('sentence_transformers'), 'SentenceTransformer'),
        reason="SentenceTransformer not available"
    )
    def test_model_encode_works(self):
        """Test that model can encode text."""
        model = get_model()
        result = model.encode(["test sentence"])
        assert result is not None
        assert len(result.shape) == 2


class TestEmbedChunks:
    """Tests for the embed_chunks function."""

    def test_embed_single_chunk(self, sample_chunks):
        """Test embedding a single chunk."""
        embeddings = embed_chunks([sample_chunks[0]])
        assert embeddings is not None
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == 1

    def test_embed_multiple_chunks(self, sample_chunks):
        """Test embedding multiple chunks."""
        embeddings = embed_chunks(sample_chunks)
        assert embeddings.shape[0] == len(sample_chunks)

    def test_embed_returns_array(self, sample_chunks):
        """Test that embeddings are returned as numpy array."""
        embeddings = embed_chunks(sample_chunks)
        assert hasattr(embeddings, 'shape')
        assert hasattr(embeddings, '__iter__')

    def test_embed_consistency(self, sample_chunks):
        """Test that same input produces same embeddings."""
        embeddings1 = embed_chunks(sample_chunks[:1])
        embeddings2 = embed_chunks(sample_chunks[:1])
        import numpy as np
        assert np.allclose(embeddings1, embeddings2)

    def test_embed_dimensions(self, sample_chunks):
        """Test that embedding dimensions are consistent."""
        model = get_model()
        expected_dim = model.get_sentence_embedding_dimension()

        embeddings = embed_chunks(sample_chunks[:1])
        assert embeddings.shape[1] == expected_dim


class TestEmbedChunksEdgeCases:
    """Edge case tests for embed_chunks."""

    def test_embed_empty_list(self):
        """Test embedding an empty list of chunks."""
        embeddings = embed_chunks([])
        assert len(embeddings) == 0

    def test_embed_very_long_chunk(self):
        """Test embedding a very long chunk."""
        long_text = " ".join(["word"] * 1000)
        embeddings = embed_chunks([long_text])
        assert embeddings.shape[0] == 1

    def test_embed_very_short_chunk(self):
        """Test embedding a single word."""
        embeddings = embed_chunks(["word"])
        assert embeddings.shape[0] == 1

    def test_embed_unicode_chunks(self):
        """Test embedding chunks with unicode characters."""
        unicode_chunks = [
            "Hello 你好",
            "مرحبا بالعالم",
            "こんにちは 世界"
        ]
        embeddings = embed_chunks(unicode_chunks)
        assert embeddings.shape[0] == 3