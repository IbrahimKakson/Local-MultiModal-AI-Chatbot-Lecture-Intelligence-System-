"""Manual Interactive Chat - Week 8 Testing.
Demonstrates the BEFORE (raw LLM) vs AFTER (full RAG chain with memory).
Type 'quit' to exit.
"""
import time

def run_before_demo():
    """BEFORE (Week 7): Raw LLM with hardcoded context, no memory."""
    from app.services.llm_engine import load_model, generate_answer_from_model

    print("=" * 60)
    print("  BEFORE (Week 7): Raw LLM - No database, no memory")
    print("=" * 60)
    print("\nLoading Phi-3...")
    load_model()
    print("Ready!\n")

    fake_context = [
        "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "The mitochondria is the powerhouse of the cell.",
    ]

    # Turn 1
    q1 = "What is photosynthesis?"
    print(f"You: {q1}")
    print("Thinking...")
    start = time.time()
    a1 = generate_answer_from_model(query=q1, context=fake_context)
    print(f"AI: {a1}")
    print(f"({time.time() - start:.1f}s)\n")

    # Turn 2 - follow up (the LLM has NO idea what we just talked about)
    q2 = "Can you explain more about that?"
    print(f"You: {q2}")
    print("Thinking...")
    start = time.time()
    a2 = generate_answer_from_model(query=q2, context=fake_context)
    print(f"AI: {a2}")
    print(f"({time.time() - start:.1f}s)\n")

    print("[!] Notice: The AI has NO memory. It doesn't know what")
    print("    'that' refers to. Each question is completely isolated.\n")


def run_after_demo():
    """AFTER (Week 8): Full RAG chain with database search + memory."""
    from app.services.vector_store import VectorStoreService
    from app.services.search_service import SearchService
    from app.services.rag_chain import RAGChain

    print("=" * 60)
    print("  AFTER (Week 8): Full RAG Chain - Database + Memory")
    print("=" * 60)

    # Step 1: Seed the database with lecture content
    print("\n[Setup] Loading lecture content into vector database...")
    vs = VectorStoreService(collection_name="week8_demo")
    vs.add_documents(
        texts=[
            "Photosynthesis is the process by which green plants convert sunlight into chemical energy stored as glucose.",
            "The mitochondria is the powerhouse of the cell, responsible for producing ATP through cellular respiration.",
            "DNA stands for deoxyribonucleic acid and carries the genetic instructions for all living organisms.",
            "Newton's first law states that an object at rest stays at rest unless acted on by an external force.",
            "Machine learning is a subset of artificial intelligence focused on learning patterns from data.",
        ],
        metadatas=[
            {"source": "biology_101.pdf", "page": 1},
            {"source": "biology_101.pdf", "page": 2},
            {"source": "biology_101.pdf", "page": 3},
            {"source": "physics.pdf", "page": 1},
            {"source": "ai_lecture.pdf", "page": 1},
        ],
        ids=["bio1", "bio2", "bio3", "phy1", "ai1"],
    )
    print("[Setup] Done! 5 lecture chunks loaded.\n")

    # Step 2: Build the chain
    print("[Setup] Building RAG chain...")
    rag = RAGChain(top_k=3)
    print("[Setup] Ready!\n")

    # Turn 1
    q1 = "What is photosynthesis?"
    print(f"You: {q1}")
    print("Thinking... (searching database + generating answer)")
    start = time.time()
    result1 = rag.ask(q1)
    print(f"AI: {result1['answer']}")
    sources = [d.metadata.get("source", "?") for d in result1["source_documents"]]
    print(f"Sources: {sources}")
    print(f"({time.time() - start:.1f}s)\n")

    # Turn 2 - follow up (the chain REMEMBERS what we just discussed)
    q2 = "Can you explain more about that?"
    print(f"You: {q2}")
    print("Thinking... (using memory + searching + generating)")
    start = time.time()
    result2 = rag.ask(q2)
    print(f"AI: {result2['answer']}")
    print(f"({time.time() - start:.1f}s)\n")

    print("[!] Notice: The AI REMEMBERS the previous question.")
    print("    It knows 'that' refers to photosynthesis because")
    print("    the chat history is injected into every prompt.\n")

    # Cleanup
    vs.chroma_client.delete_collection("week8_demo")
    print("[Cleanup] Test collection removed.")


if __name__ == "__main__":
    print("\nThis demo runs the SAME two questions through both systems")
    print("so you can see exactly what changed between Week 7 and Week 8.\n")

    run_before_demo()

    print("\n" + "-" * 60)
    print("  Now running the SAME questions through the NEW system...")
    print("-" * 60 + "\n")

    run_after_demo()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  BEFORE: You manually pass context. No memory. No search.")
    print("  AFTER:  The chain searches the DB automatically,")
    print("          picks the best chunks, feeds them to Phi-3,")
    print("          AND remembers what you said before.")
    print("=" * 60)
