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
    
    # 1. Test Vector Store alone (Often struggles with single abstract IDs like "CS-101" vs general context)
    vector_results = vector_store.query_similarity(query, top_k=3)
    
    # 2. Test Keyword Search alone
    keyword_results = search_service.keyword_search(query, top_k=3)
    
    # 3. Test Hybrid
    hybrid_results = search_service.hybrid_search(query, top_k=1)
    
    # BM25 should easily find doc2
    assert any(res["id"] == "doc2" for res in keyword_results)
    
    # Hybrid must return doc2 as the absolute #1 best result
    assert hybrid_results[0]["id"] == "doc2"
