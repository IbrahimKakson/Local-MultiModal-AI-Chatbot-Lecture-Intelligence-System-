"""End-to-End test for the LangChain RAG orchestration.
Run with: python -m tests.test_e2e_rag
"""
import time
from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService
from app.services.rag_chain import RAGChain


def main():
    print("=" * 60)
    print("  E2E Test: LangChain RAG Orchestration")
    print("=" * 60)

    # Step 1: Seed the vector database with fake lecture content
    print("\n[1/4] Seeding vector database with sample lecture chunks...")
    vs = VectorStoreService(collection_name="e2e_test")
    vs.add_documents(
        texts=[
            "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
            "The mitochondria is the powerhouse of the cell, responsible for producing ATP.",
            "DNA stands for deoxyribonucleic acid. It carries genetic instructions.",
            "The water cycle involves evaporation, condensation, and precipitation.",
            "Newton's first law states that an object at rest stays at rest unless acted on by a force.",
        ],
        metadatas=[
            {"source": "biology_101.pdf", "page": 1},
            {"source": "biology_101.pdf", "page": 2},
            {"source": "biology_101.pdf", "page": 3},
            {"source": "geography.pdf", "page": 1},
            {"source": "physics.pdf", "page": 1},
        ],
        ids=["bio1", "bio2", "bio3", "geo1", "phy1"],
    )
    print("   Done. 5 chunks inserted.")

    # Step 2: Build the RAG Chain
    print("\n[2/4] Building RAG chain (loading Llama-3.2 LLM)...")
    start = time.time()
    rag = RAGChain(top_k=3, collection_name="e2e_test")
    print(f"   Chain ready in {time.time() - start:.2f}s.")

    # Step 3: Ask a question (first turn)
    print("\n[3/4] Asking first question...")
    q1 = "What does the mitochondria do?"
    print(f"   Q: {q1}")
    start = time.time()
    result1 = rag.ask(q1)
    print(f"   A: {result1['answer']}")
    print(f"   Sources: {[d.metadata.get('source', '?') for d in result1['source_documents']]}")
    print(f"   Time: {time.time() - start:.2f}s")

    # Step 4: Ask a follow-up (tests conversational memory)
    print("\n[4/4] Asking follow-up question (testing memory)...")
    q2 = "Can you tell me more about that topic?"
    print(f"   Q: {q2}")
    start = time.time()
    result2 = rag.ask(q2)
    print(f"   A: {result2['answer']}")
    print(f"   Time: {time.time() - start:.2f}s")

    print("\n" + "=" * 60)
    print("  E2E Test Complete!")
    print("=" * 60)

    # Cleanup: remove the test collection
    vs.chroma_client.delete_collection("e2e_test")
    print("Test collection cleaned up.")


if __name__ == "__main__":
    main()
