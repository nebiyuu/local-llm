import time
import requests
import json
from ..config import settings


def ask_ollama(context: str, question: str):
    url = settings.ollama_url
    payload = {
        "model": settings.ollama_model,
        "prompt": f"""Based on the provided context, answer the user's question as accurately as possible.
If the context doesn't contain the specific information needed, try to provide a helpful response based on what is available.
Only say "I don't know" if the context is completely irrelevant to the question.

Context:
    {context}
User request:
    {question}
    """,
        "stream": True,
        "think": False,
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

                print(chunk, end="", flush=True)

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