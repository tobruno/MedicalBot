import ollama
import re
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

# from qdrant_setup import q_client, COLLECTION_NAME

# --- DATABASE SETUP ---
# Initialize Qdrant Client (Ensure the path matches your Phase 2 setup)
QDRANT_PATH = "./recursive_contextualized_qdrant"
COLLECTION_NAME = "medical_contextualized"
q_client = QdrantClient(path=QDRANT_PATH)

# Initialize both Dense and Sparse models for BGE-M3
print("⏳ Loading BGE-M3 models for Hybrid Retrieval...")
dense_model = TextEmbedding(model_name="BAAI/bge-m3")
sparse_model = SparseTextEmbedding(model_name="BAAI/bge-m3")


def generate_hyde_document(user_query):
    """
    HyDE 2.0: Uses Task-based formatting to bridge the lexical gap.
    """
    prompt = f"""
        <role>Expert Medical Encyclopedia Author</role>
        <task>Generate a hypothetical, formal medical encyclopedia excerpt that directly answers the patient's query. This text will be converted into a dense vector for database retrieval.</task>

        <patient_query>{user_query}</patient_query>

        <constraints>
        - Length: 50-80 words.
        - Tone: Academic, objective, highly professional medical terminology.
        - Output Format: Provide ONLY the raw hypothetical text. Do not use conversational filler, greetings, or include any XML tags in your response.
        </constraints>
    """
    response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()



def search_qdrant(query_text, limit=3):
    """
    Helper function to perform a true Hybrid Search (Dense + Sparse) 
    using Reciprocal Rank Fusion (RRF) in Qdrant.
    """
    # 1. Vectorize the text using both embedding strategies
    dense_vec = list(dense_model.embed([query_text]))[0].tolist()
    sparse_vec_obj = list(sparse_model.embed([query_text]))[0].as_object()
    
    # Format the sparse vector for the Qdrant API
    sparse_vec = models.SparseVector(
        indices=sparse_vec_obj.get("indices", []),
        values=sparse_vec_obj.get("values", [])
    )
    
    # 2. Execute Hybrid Query using Prefetch and RRF
    # We fetch a larger pool of candidates (limit * 2) during the prefetch phase 
    # to ensure high-quality fusion, then return only the exact requested limit.
    try:
        response = q_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=limit * 2
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=limit * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )
        
        # 3. Parse Qdrant PointStructs into a clean list of dictionaries
        results = []
        for point in response.points:
            results.append({
                "text": point.payload.get("text", "Missing text data."),
                "metadata": point.payload.get("metadata", {})
            })
            
        return results

    except Exception as e:
        print(f"❌ Database Retrieval Error: {e}")
        return []

def recursive_retrieval(user_query, max_depth=2):
    """
    Recursive Retrieval 2.0: Utilizes the Scratchpad (<thinking>) technique.
    """
    print("🔍 Running HyDE transformation...")
    hypothetical_doc = generate_hyde_document(user_query)
    
    print("📚 Performing initial database retrieval...")
    current_chunks = search_qdrant(hypothetical_doc, limit=3)
    all_retrieved_chunks = list(current_chunks)
    
    depth = 1
    while depth < max_depth:
        context_texts = "\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(current_chunks)])
        
        evaluation_prompt = f"""
            <role>Medical Retrieval Evaluator</role>
            <objective>Determine if the retrieved context contains sufficient clinical information to fully answer the primary question.</objective>

            <primary_question>{user_query}</primary_question>
            <retrieved_context>
            {context_texts}
            </retrieved_context>

            <instructions>
            1. Briefly analyze the context against the question inside <thinking> tags.
            2. If the answer is fully present in the context, output exactly: <decision>COMPLETE</decision>
            3. If critical information is missing, output <decision>SEARCH_QUERY</decision> where SEARCH_QUERY is a highly specific 2-4 word English medical keyword phrase to fetch the missing data.
            </instructions>
            """
        eval_response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': evaluation_prompt}])
        raw_output = eval_response['message']['content']
        
        # Modern parsing using Regex
        decision_match = re.search(r"<decision>(.*?)</decision>", raw_output, re.IGNORECASE | re.DOTALL)
        next_query = decision_match.group(1).strip() if decision_match else "COMPLETE"
        
        if "COMPLETE" in next_query.upper():
            print(f"✅ Gathered a complete chain of clinical evidence (Depth: {depth}).")
            break
            
        print(f"🔄 Data missing. Triggering sub-query: '{next_query}'...")
        new_chunks = search_qdrant(next_query, limit=2)
        all_retrieved_chunks.extend(new_chunks)
        current_chunks = new_chunks 
        depth += 1