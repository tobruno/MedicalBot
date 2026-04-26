import ollama
import re
import os
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel
from ontology_linker import extract_umls_cuis 

from groq import Groq 
from dotenv import load_dotenv

load_dotenv()
groq_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_key)

class MedicalRetriever:
    """
    Experimental engine allowing side-by-side comparison of 
    standard Semantic Reranking vs. Ontologically Boosted Reranking,
    combined with Recursive Context Retrieval.
    """
    def __init__(self, qdrant_path="./medical_qdrant_db", collection_name="semantic_contextualized"):
        print("⚙️ Initializing Experimental Medical Retriever...")
        self.q_client = QdrantClient(path=qdrant_path)
        self.collection = collection_name
        
        print("⏳ Loading Official BGE-M3 Model (8192 Token Limit)...")
        self.bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        
        print("⏳ Loading Cross-Encoder for Reranking...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024) 

    def retrieve(self, user_query, use_ontology=True, top_k=5, max_depth=2):
        """
        Main recursive retrieval pipeline.
        """
        print(f"\n🔍 [RETRIEVAL] Query: '{user_query}' | Ontology Boost: {use_ontology}")
        
        # 1. Transform query using HyDE
        hyde_doc = self._generate_hyde(user_query)
        
        # 2. Extract UMLS Concepts from the original patient query (only once)
        query_cuis = extract_umls_cuis(user_query) if use_ontology else set()
        
        # 3. Initial Fetch and Rerank
        current_chunks = self._fetch_and_rerank(
            search_query=hyde_doc, 
            eval_query=user_query, 
            query_cuis=query_cuis, 
            use_ontology=use_ontology, 
            fetch_limit=20, 
            top_k=top_k
        )
        
        all_retrieved_chunks = list(current_chunks)
        depth = 1
        
        # 4. Recursive Evaluation Loop
        while depth < max_depth:
            next_query = self._evaluate_context(user_query, current_chunks)
            
            if next_query == "COMPLETE":
                print(f"✅ [RETRIEVAL] Gathered a complete chain of clinical evidence (Depth: {depth}).")
                break
                
            print(f"🔄 [RETRIEVAL] Data missing. Triggering sub-query: '{next_query}'...")
            
            # Fetch additional chunks using the LLM-generated sub-query
            new_chunks = self._fetch_and_rerank(
                search_query=next_query, 
                eval_query=user_query, 
                query_cuis=query_cuis, 
                use_ontology=use_ontology, 
                fetch_limit=15, 
                top_k=2 # We fetch fewer chunks in subsequent loops to avoid context overflow
            )
            
            all_retrieved_chunks.extend(new_chunks)
            current_chunks = new_chunks 
            depth += 1
            
        return all_retrieved_chunks

    def _fetch_and_rerank(self, search_query, eval_query, query_cuis, use_ontology, fetch_limit, top_k):
        """Helper method to handle the repetitive Hybrid Search + Reranking logic."""
        # A. Hybrid Search
        candidates = self._hybrid_search(search_query, limit=fetch_limit)
        if not candidates:
            return []
            
        # B. Reranking Logic (Cross-Encoder)
        sentence_pairs = [[eval_query, chunk["text"]] for chunk in candidates]
        scores = self.reranker.predict(sentence_pairs)
        
        for idx, chunk in enumerate(candidates):
            score = float(scores[idx])
            
            if use_ontology:
                chunk_cuis = set(chunk["metadata"].get("umls_cuis", []))
                overlap = query_cuis.intersection(chunk_cuis)
                score = score * (1.0 + (len(overlap) * 0.15))
                chunk["umls_overlap"] = list(overlap)
            
            chunk["final_score"] = score

        ranked_candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        return ranked_candidates[:top_k]

    def _hybrid_search(self, query_text, limit):
        embeddings = self.bge_m3.encode(
            [query_text], 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        dense_vec = embeddings['dense_vecs'][0].tolist()
        
        # 2. Format lexical weights for Qdrant Sparse format
        sparse_dict = embeddings['lexical_weights'][0]
        sparse_indices = [int(k) for k in sparse_dict.keys()]
        sparse_values = [float(v) for v in sparse_dict.values()]
        
        sparse_vec = models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
        try:
            response = self.q_client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vec, using="dense", limit=limit * 2),
                    models.Prefetch(query=sparse_vec, using="sparse", limit=limit * 2)
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit
            )
            return [{"text": p.payload.get("text", ""), "metadata": p.payload.get("metadata", {})} for p in response.points]
        except Exception as e:
            print(f"❌ Database Retrieval Error: {e}")
            return []

    def _generate_hyde(self, query):
        prompt = f"""
            <role>Expert Medical Encyclopedia Author</role>
            <task>Generate a hypothetical, formal medical encyclopedia excerpt answering the patient's query. Output ONLY the raw text.</task>
            <patient_query>{query}</patient_query>
            """
        try:
            #response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
            response =  client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{'role': 'user', 'content': prompt}], temperature=0.3)
            #return response['message']['content'].strip()
            return response.choices[0].message.content.strip()
        except Exception:
            return query

    def _evaluate_context(self, original_query, chunks):
        """Self-Reflection LLM evaluator to trigger recursive searches."""
        context_texts = "\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])
        
        eval_prompt = f"""
            <role>Medical Retrieval Evaluator</role>
            <objective>Determine if the context contains sufficient information to answer the primary question.</objective>
            <primary_question>{original_query}</primary_question>
            <retrieved_context>\n{context_texts}\n</retrieved_context>
            <instructions>
            1. Briefly analyze the context against the question inside <thinking> tags.
            2. If complete, output exactly: <decision>COMPLETE</decision>
            3. If information is missing, output <decision>SEARCH_QUERY</decision> with a specific 2-4 word English search phrase.
            </instructions>
            """
        try:
            # response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': eval_prompt}])
            # raw_output = response['message']['content']
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[{'role': 'user', 'content': eval_prompt}],
                temperature=0.0 # Ewaluator musi być deterministyczny
            )
            raw_output = response.choices[0].message.content

            match = re.search(r"<decision>(.*?)</decision>", raw_output, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else "COMPLETE"
        except Exception:
            return "COMPLETE" # Fallback to stop recursion if LLM fails