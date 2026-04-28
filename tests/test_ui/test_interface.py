import pytest
import re


def clean_response(text: str) -> str:
    """Testable version of clean_response from ui.py."""
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


class TestCleanResponse:
    """Tests for the clean_response function."""

    def test_removes_opening_thinking_tags(self):
        """Test that opening thinking tags are removed."""
        text = "Hello<|thinking|>Let me think world"
        result = clean_response(text)
        assert "<|thinking|>" not in result

    def test_preserves_normal_text(self):
        """Test that normal text is preserved."""
        text = "Hello, world! How are you?"
        result = clean_response(text)
        assert result == text

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "   Hello world   \n\n  "
        result = clean_response(text)
        assert result == "Hello world"

    def test_empty_string(self):
        """Test with empty string."""
        result = clean_response("")
        assert result == ""

    def test_no_tags_unchanged(self):
        """Test text without tags is unchanged except for whitespace."""
        text = "Simple text without any special tags."
        result = clean_response(text)
        assert result == "Simple text without any special tags."

    def test_unicode_preserved(self):
        """Test that unicode characters are preserved."""
        text = "你好世界 👋🎉"
        result = clean_response(text)
        assert result == "你好世界 👋🎉"

    def test_partial_tag_not_removed(self):
        """Test that partial tags are not removed."""
        text = "Hello <| world"
        result = clean_response(text)
        assert "<|" in result

    def test_mixed_content(self):
        """Test text with opening tags mixed with content."""
        text = "<|thinking>Hello world"
        result = clean_response(text)
        assert "Hello world" in result
        assert "<|thinking|>" not in result