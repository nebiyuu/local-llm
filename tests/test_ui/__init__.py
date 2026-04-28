import pytest
import re


def clean_response(text: str) -> str:
    """Testable version of clean_response from ui.py."""
    text = re.sub(r"<\|[^|]+\|>", "", text)
    return text.strip()


class TestCleanResponse:
    """Tests for the clean_response function."""

    def test_removes_thinking_tags(self):
        """Test that thinking tags are removed."""
        text = "Hello<|thinking|>Let me think</|thinking|>world"
        result = clean_response(text)
        assert "<|" not in result
        assert "|>" not in result

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

    def test_multiple_thinking_tags(self):
        """Test removal of multiple thinking tags."""
        text = "<|thought>First</|thought>Hello<|thought>Second</|thought>World"
        result = clean_response(text)
        assert "Hello" in result
        assert "World" in result
        assert "<|" not in result

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


class TestCleanResponseEdgeCases:
    """Edge case tests for clean_response."""

    def test_partial_tag_not_removed(self):
        """Test that partial tags are not removed."""
        text = "Hello <| world"
        result = clean_response(text)
        assert "<|" in result

    def test_only_tag(self):
        """Test text that is only a tag."""
        text = "<|thinking>"
        result = clean_response(text)
        assert text.strip() == ""

    def test_nested_tags(self):
        """Test text with nested-like patterns."""
        text = "<|outer><|inner>content</|inner></|outer>"
        result = clean_response(text)
        assert "content" in result
        assert "<|" not in result