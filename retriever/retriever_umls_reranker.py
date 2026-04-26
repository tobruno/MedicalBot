import ollama
import re
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder
from ontology_linker import extract_umls_cuis 

class ExperimentalMedicalRetriever:
    """
    Experimental engine allowing side-by-side comparison of 
    standard Semantic Reranking vs. Ontologically Boosted Reranking.
    """
    def __init__(self, qdrant_path="./recursive_contextualized_qdrant"):
        self.q_client = QdrantClient(path=qdrant_path)
        self.collection = "medical_contextualized"
        self.dense_model = TextEmbedding(model_name="BAAI/bge-m3")
        self.sparse_model = SparseTextEmbedding(model_name="BAAI/bge-m3")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def retrieve(self, user_query, use_ontology=True, top_k=5):
        """
        Retrieves chunks with an optional Ontological Boost.
        Set use_ontology=False to see the 'Base RAG' results.
        """
        hyde_doc = self._generate_hyde(user_query)
        candidates = self._hybrid_search(hyde_doc, limit=20)
        
        # 🧠 Get UMLS Concepts only if needed
        query_cuis = extract_umls_cuis(user_query) if use_ontology else set()
        
        # 🎯 Reranking Logic
        sentence_pairs = [[user_query, chunk["text"]] for chunk in candidates]
        scores = self.reranker.predict(sentence_pairs)
        
        for idx, chunk in enumerate(candidates):
            score = float(scores[idx])
            
            if use_ontology:
                chunk_cuis = set(chunk["metadata"].get("umls_cuis", []))
                overlap = query_cuis.intersection(chunk_cuis)
                # Apply 15% boost per shared medical concept
                score = score * (1.0 + (len(overlap) * 0.15))
                chunk["umls_overlap"] = list(overlap)
            
            chunk["final_score"] = score

        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]

    def _hybrid_search(self, query, limit):
        # Implementation of Hybrid Search using RRF (same as before)
        pass

    def _generate_hyde(self, query):
        # Implementation of HyDE (same as before)
        pass