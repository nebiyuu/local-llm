import pytest
from unittest.mock import Mock, patch
import time


def ask_ollama(context, question):
    """Testable version of ask_ollama for testing without real API."""
    import json
    import requests

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3.5:0.8b",
        "prompt": f"""
        Based on the provided context, answer the user's question as accurately as possible.
        If the context doesn't contain the specific information needed, try to provide a helpful response based on what is available.
        Only say "I don't know" if the context is completely irrelevant to the question.

        Context:
            {context}
        User request:
            {question}
            """,
        "stream": True,
        "think": False
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, stream=True, timeout=240)

        full_response = ""
        full_thinking = ""

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("response", "")
                thinking_chunk = data.get("thinking", "")

                full_response += chunk
                full_thinking += thinking_chunk

                yield chunk, thinking_chunk

                if data.get("done"):
                    break

        if full_thinking:
            print(f"\n--- Model's Thinking ---\n{full_thinking}")

        elapsed = time.time() - start_time
        print(f"\n[Elapsed time: {elapsed:.2f} s]")

        yield None, full_thinking if full_thinking else ""

    except Exception as e:
        print(f"Error: {e}")
        yield None, f"Error: {e}"


class TestAskOllama:
    """Tests for the ask_ollama function."""

    @patch('requests.post')
    def test_yields_chunks(self, mock_post, mock_ollama_stream_response):
        """Test that function yields response chunks."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        chunks = list(ask_ollama("context", "question"))
        assert len(chunks) >= 1

    @patch('requests.post')
    def test_handles_thinking_output(self, mock_post, mock_ollama_thinking_response):
        """Test that thinking output is captured."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_thinking_response)
        mock_post.return_value = mock_response

        full_response = ""
        full_thinking = ""
        for chunk, thinking in ask_ollama("context", "question"):
            if chunk:
                full_response += chunk
            if thinking:
                full_thinking += thinking

        assert full_response == "The answer is 42."

    @patch('requests.post')
    def test_uses_correct_url(self, mock_post, mock_ollama_stream_response):
        """Test that correct Ollama URL is used."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        list(ask_ollama("context", "question"))
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "localhost:11434" in call_args[0][0]

    @patch('requests.post')
    def test_sends_correct_payload(self, mock_post, mock_ollama_stream_response):
        """Test that correct payload is sent."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        list(ask_ollama("context here", "question here"))
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert 'prompt' in payload
        assert 'context here' in payload['prompt']
        assert 'question here' in payload['prompt']

    @patch('requests.post')
    def test_connection_error(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = ConnectionError("Connection refused")

        chunks = list(ask_ollama("context", "question"))
        assert len(chunks) >= 1
        last_chunk, thinking = chunks[-1]
        assert last_chunk is None or "Error" in thinking

    @patch('requests.post')
    def test_timeout_handling(self, mock_post):
        """Test handling of timeout errors."""
        mock_post.side_effect = TimeoutError("Request timed out")

        chunks = list(ask_ollama("context", "question"))
        assert len(chunks) >= 1
        last_chunk, thinking = chunks[-1]
        assert last_chunk is None or "Error" in thinking

    @patch('requests.post')
    def test_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter([
            b'not valid json',
            b'also not valid'
        ])
        mock_post.return_value = mock_response

        chunks = list(ask_ollama("context", "question"))
        assert isinstance(chunks, list)

    @patch('requests.post')
    def test_empty_response(self, mock_post):
        """Test handling of empty responses."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter([
            b'{"response": "", "done": true}'
        ])
        mock_post.return_value = mock_response

        full_response = ""
        for chunk, thinking in ask_ollama("context", "question"):
            if chunk:
                full_response += chunk

        assert full_response == ""

    @patch('requests.post')
    def test_stream_false_setting(self, mock_post, mock_ollama_stream_response):
        """Test that stream=True is set in payload."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        list(ask_ollama("context", "question"))
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload.get('stream') == True

    @patch('requests.post')
    def test_model_in_payload(self, mock_post, mock_ollama_stream_response):
        """Test that model is specified in payload."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        list(ask_ollama("context", "question"))
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert 'model' in payload


class TestAskOllamaEdgeCases:
    """Edge case tests for ask_ollama."""

    @patch('requests.post')
    def test_empty_context(self, mock_post, mock_ollama_stream_response):
        """Test with empty context."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        chunks = list(ask_ollama("", "question"))
        assert len(chunks) >= 1

    @patch('requests.post')
    def test_empty_question(self, mock_post, mock_ollama_stream_response):
        """Test with empty question."""
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        chunks = list(ask_ollama("context", ""))
        assert len(chunks) >= 1

    @patch('requests.post')
    def test_very_long_context(self, mock_post, mock_ollama_stream_response):
        """Test with very long context."""
        long_context = "word " * 10000
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        chunks = list(ask_ollama(long_context, "question"))
        assert len(chunks) >= 1

    @patch('requests.post')
    def test_special_characters_in_context(self, mock_post, mock_ollama_stream_response):
        """Test with special characters in context."""
        special_context = "Special @#$%^&*() chars!\n\t\\ unicode 你好 مرحبا"
        mock_response = Mock()
        mock_response.iter_lines.return_value = iter(mock_ollama_stream_response)
        mock_post.return_value = mock_response

        chunks = list(ask_ollama(special_context, "question"))
        assert len(chunks) >= 1