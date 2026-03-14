import pytest
import os
import shutil
from app.core.config import settings
from app.services.vector_store import VectorStoreService

@pytest.fixture(scope="module")
def persistent_db_dir():
    test_dir = os.path.join(settings.data_dir, "test_chroma_db")
    settings.chroma_dir = test_dir
    yield test_dir
    # Cleanup after test module finishes
    if os.path.exists(test_dir):
        # A tiny delay or retry might be needed if SQLite locks it but we assume clean teardown
        shutil.rmtree(test_dir, ignore_errors=True)

def test_vector_store_persistence(persistent_db_dir):
    service = VectorStoreService(collection_name="test_collection")
    
    texts = [
        "The sky is blue today.",
        "Dogs are great pets.",
        "The quick brown fox jumps over the lazy dog."
    ]
    metadatas = [
        {"source": "doc1", "page": 1},
        {"source": "doc2", "page": 1},
        {"source": "doc3", "page": 1}
    ]
    ids = ["id1", "id2", "id3"]
    
    # Add documents
    service.add_documents(texts, metadatas, ids)
    
    # Query for something related to the sky
    results = service.query_similarity("What color is the sky?", top_k=1)
    
    assert len(results) == 1
    assert "blue" in results[0]["text"].lower()
    assert results[0]["id"] == "id1"
    
    # Query for something related to dogs
    results_dogs = service.query_similarity("Tell me about puppies.", top_k=1)
    
    assert len(results_dogs) == 1
    assert "dogs" in results_dogs[0]["text"].lower()
    
    # Test persistence by re-instantiating the service completely (loading from disk)
    new_service = VectorStoreService(collection_name="test_collection")
    persistent_results = new_service.query_similarity("What color is the sky?", top_k=1)
    
    assert len(persistent_results) == 1
    assert persistent_results[0]["text"] == "The sky is blue today."
