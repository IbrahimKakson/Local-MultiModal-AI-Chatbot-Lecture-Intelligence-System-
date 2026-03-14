import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class VectorStoreService:
    def __init__(self, collection_name: str = "lecture_chunks"):
        """Initialize ChromaDB and SentenceTransformer embedding model."""
        os.makedirs(settings.chroma_dir, exist_ok=True)
        # Initialize chroma client with persistence directory
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
        
        # We use a lightweight open-source embedding model
        # SentenceTransformer('all-MiniLM-L6-v2') creates 384-dimensional embeddings
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Get or create the collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Embed and add a list of text chunks precisely to the database."""
        if not texts:
            return

        # 1. Convert texts into mathematical numbers (embeddings)
        embeddings = self.encoder.encode(texts).tolist()

        # 2. Add precisely to the database
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def query_similarity(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the database for chunks mathematically similar to the query."""
        # 1. Embed the query
        query_embedding = self.encoder.encode([query]).tolist()

        # 2. Query the database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # 3. Format results into a simpler list of dicts
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    "id": results["ids"][0][i]
                }
                formatted_results.append(doc)
                
        return formatted_results

# Export standard stubs that rag_chain expects currently
def init_vector_store():
    """Stub from older code, returning instance"""
    return VectorStoreService()

def persist_vectors():
    """Stub from older code"""
    pass
