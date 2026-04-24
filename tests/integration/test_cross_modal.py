import pytest
import os
import shutil
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.search_service import SearchService

@pytest.fixture(scope="module")
def cross_modal_db_dir():
    original_chroma_dir = settings.chroma_dir
    test_dir = os.path.join(settings.data_dir, "test_cross_modal_db")
    settings.chroma_dir = test_dir
    try:
        yield test_dir
    finally:
        settings.chroma_dir = original_chroma_dir
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)

def test_cross_modal_retrieval(cross_modal_db_dir):
    """Phase 2, Step 3: Cross-Modal Test"""
    vector_store = VectorStoreService(collection_name="cross_modal_test_collection")
    search_service = SearchService(vector_store)
    
    # We will simulate injecting data from both PDF and Audio
    texts = [
        "The mitochondria is the powerhouse of the cell, generating ATP.", # from PDF
        "As we discussed in the lecture, the mitochondria acts as a powerhouse to produce ATP.", # from Audio
        "Photosynthesis occurs in chloroplasts." # Irrelevant
    ]
    metadatas = [
        {"type": "pdf", "source": "biology_book.pdf", "page": 10},
        {"type": "audio", "source": "lecture.mp3", "start": 12.5},
        {"type": "pdf", "source": "biology_book.pdf", "page": 15}
    ]
    ids = ["pdf_chunk_1", "audio_chunk_1", "pdf_chunk_2"]
    
    # Add documents to the vector store
    search_service.add_documents(texts, metadatas, ids)
    print(f"\n\n[CROSS-MODAL TEST]")
    print(f"Seeded 3 documents (1 audio, 2 pdfs).")
    
    # Query for the topic present in both modalities
    query = "What is the powerhouse of the cell and what does it produce?"
    print(f"Query: '{query}'\n")
    
    # Perform hybrid search
    results = search_service.hybrid_search(query, top_k=2)
    
    print("Top 2 Retrieved Sources:")
    for i, res in enumerate(results):
        meta = res["metadata"]
        src_type = meta.get("type")
        source = meta.get("source")
        detail = f"Page {meta.get('page')}" if src_type == "pdf" else f"Timestamp {meta.get('start')}s"
        print(f"  #{i+1} [Type: {src_type.upper()}] | Source: {source} ({detail}) | Text: '{res['text'][:50]}...'")
        
    assert len(results) == 2
    
    # Verify both pdf and audio sources are retrieved
    source_types = [res["metadata"].get("type") for res in results]
    assert "pdf" in source_types, "Failed to retrieve PDF source"
    assert "audio" in source_types, "Failed to retrieve Audio source"
