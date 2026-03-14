from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np

from app.services.vector_store import VectorStoreService

class SearchService:
    def __init__(self, vector_store: VectorStoreService):
        """Initialize SearchService with an existing vector store and build the BM25 index."""
        self.vector_store = vector_store
        
        # We will hold documents in memory for keyword search
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.bm25 = None
        
        # Build BM25 from existing data in ChromaDB
        self._rebuild_bm25_index()
        
    def _rebuild_bm25_index(self):
        """Pulls all documents from ChromaDB and builds an in-memory exact-match index."""
        results = self.vector_store.collection.get()
        if results and results["documents"]:
            self.documents = results["documents"]
            self.metadatas = results["metadatas"]
            self.ids = results["ids"]
            
            import re
            def tokenize(text):
                # Remove punctuation except hyphens, then split by whitespace
                return [w for w in re.sub(r'[^\w\s-]', '', text.lower()).split() if w]
                
            tokenized_corpus = [tokenize(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        else:
            self.documents = []
            self.metadatas = []
            self.ids = []
            self.bm25 = None

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Adds documents to the Vector DB and updates the BM25 index."""
        self.vector_store.add_documents(texts, metadatas, ids)
        # Rebuild the BM25 index to include the new docs
        self._rebuild_bm25_index()

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Exact keyword matching using BM25."""
        if not self.bm25:
            return []
            
        import re
        def tokenize(text):
            return [w for w in re.sub(r'[^\w\s-]', '', text.lower()).split() if w]
            
        tokenized_query = tokenize(query)
        # Get raw scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top matching document indices
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n:
            if scores[idx] > 0: # Only include if there is some keyword overlap
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx] if self.metadatas else {},
                    "score": float(scores[idx]),
                    "id": self.ids[idx]
                })
        return results

    def _reciprocal_rank_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], k=60) -> List[Dict]:
        """Combine two lists of search results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        
        # Helper to add to RRF dictionary
        def add_to_rrf(results, weight=1.0):
            for rank, item in enumerate(results):
                doc_id = item["id"]
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = {"item": item, "score": 0.0}
                # RRF Formula: 1 / (k + rank), where rank is 1-indexed
                rrf_scores[doc_id]["score"] += weight * (1.0 / (k + rank + 1))
                
        # We weight them equally (1.0) for true hybrid
        add_to_rrf(vector_results)
        add_to_rrf(keyword_results)
        
        # Sort by the combined RRF score descending
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [res["item"] for res in sorted_results]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform both Vector and Keyword search, fuse the results, and return the best."""
        # 1. Ask the AI Librarian (Semantic Math Search)
        # We fetch more than top_k to give the fusion algorithm options
        vector_results = self.vector_store.query_similarity(query, top_k=top_k * 2)
        
        # 2. Ask the Keyword Index (Ctrl+F Exact Search)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        
        # 3. Combine them using Reciprocal Rank Fusion
        fused_items = self._reciprocal_rank_fusion(vector_results, keyword_results)
        
        return fused_items[:top_k]
