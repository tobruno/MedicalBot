import ollama
import re
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import the Cross-Encoder for Reranking
from sentence_transformers import CrossEncoder

# --- DATABASE & MODEL SETUP ---
QDRANT_PATH = "./recursive_contextualized_qdrant"
COLLECTION_NAME = "medical_contextualized"
q_client = QdrantClient(path=QDRANT_PATH)

print("⏳ Loading BGE-M3 models for Hybrid Search...")
dense_model = TextEmbedding(model_name="BAAI/bge-m3")
sparse_model = SparseTextEmbedding(model_name="BAAI/bge-m3")

print("⏳ Loading Cross-Encoder Reranker...")
# We use the official companion reranker to your BGE-M3 model
reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)


def generate_hyde_document(user_query):
    """HyDE 2.0: Generates hypothetical clinical text."""
    prompt = f"""
<role>Expert Medical Encyclopedia Author</role>
<task>Generate a hypothetical, formal medical encyclopedia excerpt that directly answers the patient's query. This text will be converted into a dense vector for database retrieval.</task>

<patient_query>{user_query}</patient_query>

<constraints>
  - Length: 50-80 words.
  - Tone: Academic, objective, highly professional medical terminology.
  - Output Format: Provide ONLY the raw hypothetical text. Do not use XML tags.
</constraints>
"""
    response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()


def search_qdrant_hybrid(query_text, fetch_limit=15):
    """
    STAGE 1: High-Recall Hybrid Retrieval
    Fetches a larger pool of candidates (e.g., 15) using RRF.
    """
    dense_vec = list(dense_model.embed([query_text]))[0].tolist()
    sparse_vec_obj = list(sparse_model.embed([query_text]))[0].as_object()
    
    sparse_vec = models.SparseVector(
        indices=sparse_vec_obj.get("indices", []),
        values=sparse_vec_obj.get("values", [])
    )
    
    try:
        response = q_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=fetch_limit * 2),
                models.Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit * 2)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit
        )
        
        results = [{"text": p.payload.get("text", ""), "metadata": p.payload.get("metadata", {})} for p in response.points]
        return results
    except Exception as e:
        print(f"❌ Database Retrieval Error: {e}")
        return []


def rerank_candidates(original_query, candidates, top_k=3):
    """
    STAGE 2: High-Precision Reranking (Cross-Encoder)
    Scores the retrieved chunks based on deep semantic relevance to the query.
    """
    if not candidates:
        return []

    # Prepare pairs for the Cross-Encoder: [(query, chunk_1), (query, chunk_2), ...]
    # Note: We compare against the original user query here, not the HyDE query, 
    # to ensure the final chunks directly address the patient's actual question.
    sentence_pairs = [[original_query, chunk["text"]] for chunk in candidates]
    
    # Calculate relevance scores
    scores = reranker_model.predict(sentence_pairs)
    
    # Attach scores to candidates and sort them from highest to lowest
    for idx, chunk in enumerate(candidates):
        chunk["rerank_score"] = scores[idx]
        
    ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    
    # Return only the top_k absolute best chunks
    return ranked_candidates[:top_k]


def recursive_retrieval(user_query, max_depth=2):
    """
    Optimized Retrieval:
    - Stage 1: Fetches top 30 candidates using Hybrid/RRF (Cheap).
    - Stage 2: Reranks top 30 down to top 10 (Expensive but precise).
    - Stage 3: LLM uses Top 10 chunks for generation.
    """
    print("🔍 Running HyDE transformation...")
    hypothetical_doc = generate_hyde_document(user_query)
    
    # We fetch 30 candidates to give the Reranker enough context to work with
    print("📚 Stage 1: Fetching top 30 broad candidates from Qdrant...")
    initial_candidates = search_qdrant_hybrid(hypothetical_doc, fetch_limit=30)
    
    # We rerank those 30 and keep the Top 5 for the generator
    print("🎯 Stage 2: Reranking top 30 down to top 5 using Cross-Encoder...")
    current_chunks = rerank_candidates(original_query=user_query, candidates=initial_candidates, top_k=5)
    
    all_retrieved_chunks = list(current_chunks)
    
    depth = 1
    while depth < max_depth:
        # We pass 10 chunks to the evaluator to see if evidence is sufficient
        context_texts = "\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(current_chunks)])
        
        evaluation_prompt = f"""
            <role>Medical Retrieval Evaluator</role>
            <objective>Determine if the retrieved context (Top 10 chunks) contains sufficient information.</objective>

            <primary_question>{user_query}</primary_question>
            <retrieved_context>
            {context_texts}
            </retrieved_context>

            <instructions>
            1. Analyze the context inside <thinking> tags.
            2. Output <decision>COMPLETE</decision> or <decision>SEARCH_QUERY</decision>.
            </instructions>
            """
        
        eval_response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': evaluation_prompt}])
        raw_output = eval_response['message']['content']
        
        decision_match = re.search(r"<decision>(.*?)</decision>", raw_output, re.IGNORECASE | re.DOTALL)
        next_query = decision_match.group(1).strip() if decision_match else "COMPLETE"
        
        if "COMPLETE" in next_query.upper():
            print(f"✅ Evidence chain complete with Top 5 chunks.")
            break
            
        print(f"🔄 Data missing. Triggering sub-query for missing info...")
        new_broad = search_qdrant_hybrid(next_query, fetch_limit=20)
        # For recursive steps, we might keep only top 5 new chunks to avoid context overflow
        new_top = rerank_candidates(original_query=next_query, candidates=new_broad, top_k=5)
        
        all_retrieved_chunks.extend(new_top)
        current_chunks = new_top 
        depth += 1
        
    return all_retrieved_chunks[:5] # Ensure we don't exceed context limits