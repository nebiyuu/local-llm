import fitz  # PyMuPDF
import requests
import json
import time
from sentence_transformers import SentenceTransformer
import gc
import chromadb



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
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3.5:0.8b",
        "prompt": f"""
        if the question is not answerable based on the provided context, 
        say "I don't know" or something similar. Do not attempt to fabricate an answer.
        If you don't know, just say you don't know.
        Do not try to use the context to make up an answer if it's not there.
        
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


chroma_client = chromadb.Client()

def store_in_chroma(chunks, embeddings, filename):
    collection = chroma_client.get_or_create_collection(name="documents")
    # use filename + index as ID so multiple docs don't overwrite each other
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    return collection


def query_chroma(collection, question_embedding, n_results=3):
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


if __name__ == "__main__":
    text = extract_text_from_pdf("testpdfs/Automata Course Outline.pdf")
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    
    collection = store_in_chroma(chunks, embeddings)
    
    question = "what is the Course Code?"
    question_embedding = embed_chunks([question])
    
    relevant_chunks = query_chroma(collection, question_embedding)
    print("Relevant chunks:")
    for chunk in relevant_chunks:
        print(chunk)
        print("---")  