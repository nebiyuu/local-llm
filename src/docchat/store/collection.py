import chromadb
from chromadb.config import Settings as ChromaSettings
from ..config import settings


def get_client():
    return chromadb.Client(settings=ChromaSettings(persist_directory=settings.chroma_persist_dir, anonymized_telemetry=False))


_client = None


def get_collection():
    global _client
    if _client is None:
        _client = get_client()
    return _client.get_or_create_collection(name=settings.collection_name)


class DocumentCollection:
    def __init__(self):
        self.client = get_client()
        self.collection = self.client.get_or_create_collection(name=settings.collection_name)

    def add(self, chunks: list, embeddings: list, filename: str):
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        return self.collection

    def query(self, question_embedding, n_results: int = None):
        if n_results is None:
            n_results = settings.n_retrieval_chunks
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=n_results,
        )
        chunks = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]
        return chunks, sources

    def reset(self):
        self.collection.delete(where={})
        return self.collection