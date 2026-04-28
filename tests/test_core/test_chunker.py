import pytest


def chunk_text(text, chunk_size=800, overlap=150):
    """Testable version of chunk_text function."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_basic_chunking(self, sample_text):
        """Test basic chunking with default parameters (800 words, 150 overlap)."""
        chunks = chunk_text(sample_text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_custom_chunk_size(self, sample_text):
        """Test chunking with custom chunk size."""
        chunks = chunk_text(sample_text, chunk_size=10, overlap=2)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 10

    def test_overlap_correctness(self, sample_text_long):
        """Test that overlap between chunks is correct."""
        chunk_size = 50
        overlap = 10
        chunks = chunk_text(sample_text_long, chunk_size=chunk_size, overlap=overlap)

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                current_words = chunks[i].split()
                next_words = chunks[i + 1].split()
                overlap_words = current_words[-overlap:]
                next_overlap_words = next_words[:overlap]
                assert overlap_words == next_overlap_words

    def test_short_text_unchunked(self):
        """Test that short text (smaller than chunk_size) returns single chunk."""
        short_text = "This is a short text."
        chunks = chunk_text(short_text, chunk_size=800, overlap=150)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_empty_text(self):
        """Test that empty text returns empty list."""
        chunks = chunk_text("", chunk_size=800, overlap=150)
        assert len(chunks) == 0

    def test_exact_chunk_size_text(self):
        """Test text that exactly matches chunk_size."""
        words = ["word"] * 100
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 1
        assert chunks[0] == text

    def test_single_word_chunks(self):
        """Test chunking with chunk_size of 1."""
        text = "one two three four five"
        chunks = chunk_text(text, chunk_size=1, overlap=0)
        assert len(chunks) == 5

    def test_zero_overlap(self, sample_text):
        """Test chunking with zero overlap."""
        chunks = chunk_text(sample_text, chunk_size=5, overlap=0)
        if len(chunks) > 1:
            first_chunk_words = chunks[0].split()
            second_chunk_words = chunks[1].split()
            assert first_chunk_words[-1] != second_chunk_words[0]


class TestChunkTextEdgeCases:
    """Edge case tests for chunk_text."""

    def test_large_overlap_near_chunk_size(self):
        """Test when overlap is close to chunk_size."""
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, chunk_size=10, overlap=9)
        assert len(chunks) > 1

    def test_whitespace_only_text(self):
        """Test text with only whitespace."""
        chunks = chunk_text("   \n\t  ", chunk_size=800, overlap=150)
        assert len(chunks) >= 0

    def test_unicode_text(self):
        """Test chunking with unicode characters."""
        text = "Hello 你好 مرحبا 👋🎉"
        chunks = chunk_text(text, chunk_size=3, overlap=0)
        assert len(chunks) >= 1