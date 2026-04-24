import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService

def run_manual_test():
    print("\n" + "=" * 60)
    print("  Hybrid Search Demo: Vector Search vs Keyword Search vs Hybrid")
    print("=" * 60)

    # Step 1: Setup
    print("\n[1/4] Initializing Vector Store + Search Service...")
    vector_store = VectorStoreService(collection_name="search_demo_collection")
    search_service = SearchService(vector_store)
    print("  Done!")

    # Step 2: Add lecture-like documents
    print("\n[2/4] Adding sample lecture documents to the database...")
    texts = [
        "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "The specific identifier for the introduction course is CS-101. It is recommended for freshmen.",
        "Machine learning is a subset of artificial intelligence that learns patterns from data.",
        "Newton's second law states that Force equals mass times acceleration, or F=ma.",
        "The mitochondria is the powerhouse of the cell, responsible for producing ATP.",
    ]
    metadatas = [
        {"source": "biology.pdf", "page": 1},
        {"source": "course_catalog.pdf", "page": 5},
        {"source": "ai_lecture.pdf", "page": 1},
        {"source": "physics.pdf", "page": 3},
        {"source": "biology.pdf", "page": 2},
    ]
    ids = ["bio1", "cs1", "ai1", "phy1", "bio2"]

    search_service.add_documents(texts, metadatas, ids)
    print(f"  {len(texts)} documents added successfully!")

    # Step 3: Run a keyword-heavy query that shows the difference
    query = "CS-101"
    print(f"\n[3/4] Running searches for query: '{query}'")
    print("-" * 50)

    # Vector search alone
    print("\n  A) VECTOR SEARCH (Semantic / Meaning-based):")
    vector_results = vector_store.query_similarity(query, top_k=3)
    for i, r in enumerate(vector_results):
        print(f"     #{i+1}: [{r['id']}] \"{r['text'][:80]}...\"")
        print(f"          Distance: {r['distance']:.4f}")

    # Keyword search alone
    print("\n  B) KEYWORD SEARCH (BM25 / Exact match):")
    keyword_results = search_service.keyword_search(query, top_k=3)
    if keyword_results:
        for i, r in enumerate(keyword_results):
            print(f"     #{i+1}: [{r['id']}] \"{r['text'][:80]}...\"")
            print(f"          Score: {r['score']:.4f}")
    else:
        print("     (No keyword matches found)")

    # Hybrid search
    print("\n  C) HYBRID SEARCH (Vector + Keyword fused with RRF):")
    hybrid_results = search_service.hybrid_search(query, top_k=3)
    for i, r in enumerate(hybrid_results):
        print(f"     #{i+1}: [{r['id']}] \"{r['text'][:80]}...\"")

    # Step 4: Verify the hybrid search got it right
    print("\n" + "-" * 50)
    print(f"\n[4/4] Analysis:")
    if hybrid_results and hybrid_results[0]["id"] == "cs1":
        print("  SUCCESS! Hybrid search correctly ranked the CS-101 document as #1.")
        print("  This proves that combining semantic + keyword search gives the best results.")
    else:
        print("  The hybrid search did not rank CS-101 first. Check the fusion weights.")

    # Cleanup
    vector_store.chroma_client.delete_collection("search_demo_collection")
    print("\n  [Cleanup] Test collection removed.")
    print("=" * 60)

if __name__ == "__main__":
    run_manual_test()
