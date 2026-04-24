import pytest
import os
import shutil
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService

@pytest.fixture(scope="module")
def hybrid_db_dir():
    original_chroma_dir = settings.chroma_dir
    test_dir = os.path.join(settings.data_dir, "test_hybrid_db")
    settings.chroma_dir = test_dir
    try:
        yield test_dir
    finally:
        settings.chroma_dir = original_chroma_dir
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

def test_hybrid_search(hybrid_db_dir):
    vector_store = VectorStoreService(collection_name="hybrid_test_collection")
    search_service = SearchService(vector_store)
    
    # We will test a case where Vector Search alone might fail to highly rank
    # an exact keyword match.
    texts = [
        "This is a general paragraph describing the course introductory elements for Computer Science.",
        "The specific identifier for the introduction course is CS-101. It is recommended for freshmen.",
        "Introductory computing logic and programming basics."
    ]
    metadatas = [{"type": "general"}, {"type": "specific"}, {"type": "general"}]
    ids = ["doc1", "doc2", "doc3"]
    
    search_service.add_documents(texts, metadatas, ids)
    
    # Keyword heavy query
    query = "CS-101"
    print(f"\n\n[HYBRID SEARCH TEST]")
    print(f"Query: '{query}'")
    
    # 1. Test Vector Store alone
    vector_results = vector_store.query_similarity(query, top_k=3)
    print("\n[1] Semantic Search Results:")
    for i, res in enumerate(vector_results):
        print(f"   #{i+1}: ID={res['id']}, Distance={res.get('distance', 0):.4f} | Text: '{res['text'][:40]}...'")
    
    # 2. Test Keyword Search alone
    keyword_results = search_service.keyword_search(query, top_k=3)
    print("\n[2] BM25 Keyword Results:")
    for i, res in enumerate(keyword_results):
        print(f"   #{i+1}: ID={res['id']}, Score={res.get('score', 0):.4f} | Text: '{res['text'][:40]}...'")
    
    # 3. Test Hybrid
    hybrid_results = search_service.hybrid_search(query, top_k=1)
    print("\n[3] Hybrid Fusion Results (Top 1):")
    print(f"   🥇 ID={hybrid_results[0]['id']} | Text: '{hybrid_results[0]['text'][:60]}...'")
    
    # BM25 should easily find doc2
    assert any(res["id"] == "doc2" for res in keyword_results)
    
    # Semantic search might struggle to rank the abstract ID 'CS-101' as #1
    # but in this case all-MiniLM is smart enough to find it. The key is that 
    # Hybrid search reliably returns it as the absolute best result.
    
    # Hybrid must return doc2 as the absolute #1 best result
    assert hybrid_results[0]["id"] == "doc2"
