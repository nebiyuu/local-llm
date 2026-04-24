import fitz  # PyMuPDF
import requests
import json
import time
import os
from sentence_transformers import SentenceTransformer
import gc
import chromadb

# Global model instance
_model = None

def get_model():
    global _model
    if _model is None:
        local_model_path = "models/amharic-embed"
        if os.path.exists(local_model_path):
            model_path = local_model_path
        else:
            model_path = 'rasyosef/RoBERTa-Amharic-Embed-Medium'
        _model = SentenceTransformer(
            model_path,
            device='cpu',
            cache_folder="./model_cache"
        )
    return _model



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
        Based on the provided context, answer the user's question as accurately as possible.
        If the context doesn't contain the specific information needed, 
        try to provide a helpful response based on what is available.
        Only say "I don't know" if the context is completely irrelevant to the question.
        
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
                chunk = data.get("response", "")
                thinking_chunk = data.get("thinking", "")
                
                full_response += chunk                
                full_thinking += thinking_chunk
                
                print(chunk, end="", flush=True)  # Print response as it comes in
                
                # Yield the current chunk for streaming
                yield chunk, thinking_chunk

                if data.get("done"):
                    break

        if full_thinking:
            print(f"\n--- Model's Thinking ---\n{full_thinking}")

        elapsed = time.time() - start_time
        print(f"\n[Elapsed time: {elapsed:.2f} s]")

        # Final yield to inchdicate completion
        yield None, full_thinking if full_thinking else ""

    except Exception as e:
        print(f"Error: {e}")
        yield None, f"Error: {e}"
    
    
    
def chunk_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # overlap so chunks don't cut mid-idea
    return chunks


def embed_chunks(texts):
    model = get_model()
    embeddings = model.encode(texts)
    return embeddings


chroma_client = chromadb.PersistentClient(path="./chroma_db")

def store_in_chroma(chunks, embeddings, filename):
    collection = chroma_client.get_or_create_collection(name="documents")
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )
    return collection


def delete_from_chroma(filename):
    collection = chroma_client.get_or_create_collection(name="documents")
    try:
        collection.delete(where={"source": filename})
        return True
    except Exception as e:
        print(f"Delete error: {e}")
        return False


def get_document_content(filename):
    collection = chroma_client.get_or_create_collection(name="documents")
    try:
        results = collection.get(where={"source": filename})
        if results and results.get("documents"):
            return results["documents"]
        return []
    except Exception as e:
        print(f"Get doc error: {e}")
        return []


def get_all_documents():
    collection = chroma_client.get_or_create_collection(name="documents")
    try:
        results = collection.get()
        docs = {}
        for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            src = meta.get("source", "unknown")
            if src not in docs:
                docs[src] = {"chunks": [], "count": 0}
            docs[src]["chunks"].append(doc)
            docs[src]["count"] += 1
        return docs
    except Exception as e:
        print(f"Get all docs error: {e}")
        return {}


def query_chroma(collection, question_embedding, n_results=5):
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources  