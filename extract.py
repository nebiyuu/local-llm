import fitz  # PyMuPDF
import requests
import json
import time
from sentence_transformers import SentenceTransformer
import gc


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
        
    return full_text

# Quick test
# path = "testpdfs/Nebiyu Essayas.pdf"
# print(extract_text_from_pdf(path))



def ask_ollama(context, question):
    url = "http://ollama:11434/api/generate"
    payload = {
        "model": "qwen3.5:0.8b",
        "prompt": f"""
        
        Context:
            {context}
        User request:
            {question}
            """,
        "stream": True,
        "think": False
    #     "options": {
    #     "repeat_penalty": 1.15,  # Prevents the loop
    #     "temperature": 0.8,      # Adds variety
    #     "top_p": 0.9,            # Keeps it coherent
    #     "repeat_last_n": 64,     # How far back it looks for repetitions
    #     "num_ctx": 1024          # Small models work better with shorter context
    # }
    }

    try:
        start_time = time.time()
        response = requests.post(url, json=payload, stream=True, timeout=240)

        full_response = ""
        full_thinking = ""

        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                full_response += data.get("response", "")                
                full_thinking += data.get("thinking", "")
                print(data.get("response", ""), end="", flush=True)  # Print response as it comes in

                if data.get("done"):
                    break

        if full_thinking:
            print(f"\n--- Model's Thinking ---\n{full_thinking}")

        elapsed = time.time() - start_time
        print(f"\n[Elapsed time: {elapsed:.2f} s]")

        return full_response, full_thinking

    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
    
    
def chunk_text(text, chunk_size=100, overlap=10):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # overlap so chunks don't cut mid-idea
    return chunks


def embed_chunks(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(texts)
    del model
    gc.collect()
    return embeddings


if __name__ == "__main__":
    # all your test code here
    # --- EXECUTION ---
    pdf_text = extract_text_from_pdf("testpdfs/Nebiyu Essayas.pdf") # Using your function from Stage 2
    user_query = "what school did he go to"  # Example question about the CV
    
    print("Thinking...")
    
    # Assuming 'pdf_text' contains your full CV text from Stage 2
    short_context = pdf_text[:1000] # Just the first 1000 characters
    
    # print(f"--- Sending a smaller chunk ({len(short_context)} chars) ---")
    
    # Now call your function with the smaller chunk
    # Using the streaming function from before is still recommended!
    answer = ask_ollama(pdf_text, user_query)
    print(f"\nOllama's Answer:\n{answer}")
    
    
    # answer = ask_ollama(pdf_text, user_query)
    # print(f"\nOllama's Answer:\n{answer}")
    