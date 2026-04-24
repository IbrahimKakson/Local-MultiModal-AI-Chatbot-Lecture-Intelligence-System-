import sys
import os

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_store import VectorStoreService

def run_manual_test():
    print("\nInitializing Vector Store (ChromaDB + SentenceTransformers)...")
    try:
        service = VectorStoreService(collection_name="manual_test_collection")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    print("Adding sample documents to the database...")
    
    texts = [
        "Python is a high-level, interpreted programming language.",
        "Artificial Intelligence is the simulation of human intelligence processes by machines.",
        "The mitochondria is the powerhouse of the cell.",
        "A vector database is designed specifically to handle the unique structure of vector embeddings."
    ]
    metadatas = [{"source": "wiki"} for _ in texts]
    ids = ["doc1", "doc2", "doc3", "doc4"]
    
    service.add_documents(texts, metadatas, ids)
    print("Documents successfully embedded and stored!\n")
    
    queries = [
        "What does mitochondria do?",
        "Tell me about vector DBs.",
        "What is AI?"
    ]
    
    for q in queries:
        print(f"Querying: '{q}'")
        results = service.query_similarity(q, top_k=1)
        if results:
            print(f"-> Closest mathematical match: '{results[0]['text']}'")
            print(f"-> Distance score: {results[0]['distance']:.4f}\n")
        else:
            print("-> No results found.\n")

if __name__ == "__main__":
    run_manual_test()
