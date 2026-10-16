import os
import re
import sys
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Langchain i RAGAS
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    ContextRelevance,
    faithfulness,
    answer_relevancy,
)
from ragas.run_config import RunConfig

# Twoje importy z silnika RAG
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retriever.decomposer import decompose_query
from retriever.retriever import MedicalRetriever
from generator import generate_grounded_response

# Ładowanie kluczy
load_dotenv()

# --- KONFIGURACJA ---
QUESTIONS_FILE = "ragas/testing/liveqa_sampled_evaluated.csv"
RESULTS_FILE = "ragas/testing/ragas_liveqa_results.csv"
SAMPLE_SIZE = 20 # Ustaw limit pytań do testów (generowanie i ocena trwa!). Ustaw na None, aby przetworzyć wszystkie.
COLLECTION_NAME = "semantic_contextualised_with_ontology_v1"  # Upewnij się, że to jest nazwa kolekcji, której używasz w Qdrant

print("🔧 Inicjalizacja silnika Qdrant...")
retriever_engine = MedicalRetriever(collection_name = COLLECTION_NAME)

import time # Dodaj to na początku pliku do mierzenia latencji

def run_medical_rag(user_complex_query, use_decomposition=True, use_recursion=True):
    """
    Uruchamia pipeline RAG z przełącznikami dla dekompozycji i rekurencji.
    Zwraca odpowiedź, konteksty oraz czas wykonania w sekundach.
    """
    start_time = time.time()
    
    # 1. Dekompozycja
    if use_decomposition:
        atomic_queries = decompose_query(user_complex_query)
        print(f"  📌 Zidentyfikowano intencje: {atomic_queries}")
    else:
        atomic_queries = [user_complex_query]
        print(f"  📌 Dekompozycja wyłączona. Używam oryginału.")
    
    # Ustawienie głębokości (1 = standardowe pobranie, >1 = rekurencja)
    current_max_depth = 2 if use_recursion else 1

    all_retrieved_chunks = []
    seen_chunk_texts = set()
    
    # 2. Wyszukiwanie
    for atomic_query in atomic_queries:
        chunks = retriever_engine.retrieve(
            user_query=atomic_query, 
            use_ontology=False, 
            top_k=3,
            max_depth=current_max_depth # <-- Podpinamy parametr rekurencji
        )
        for chunk in chunks:
            if chunk['text'] not in seen_chunk_texts:
                all_retrieved_chunks.append(chunk)
                seen_chunk_texts.add(chunk['text'])
                
    # 3. Zabezpieczenie przed brakiem danych
    if not all_retrieved_chunks:
        all_retrieved_chunks = [{"text": "Lack of medical data."}]
        
    # 4. Generacja
    raw_response = generate_grounded_response(user_complex_query, all_retrieved_chunks)
    
    if isinstance(raw_response, dict) and "answer" in raw_response:
        final_answer = raw_response["answer"]
    else:
        final_answer = str(raw_response)

    contexts_texts = [chunk['text'] for chunk in all_retrieved_chunks]
    
    end_time = time.time()
    latency = end_time - start_time

    return final_answer, contexts_texts, latency

def main():
    print(f"📊 Wczytywanie pytań z {QUESTIONS_FILE}...")
    df_original = pd.read_csv(QUESTIONS_FILE)
    
    if SAMPLE_SIZE:
        df_original = df_original.head(SAMPLE_SIZE)
        print(f"⚠️ Uwaga: Przetwarzam tylko pierwsze {SAMPLE_SIZE} pytań w celach testowych.")

    # Definiujemy scenariusze do przetestowania
    configs = [
        {"name": "Baseline", "decomp": False, "recur": False},
        {"name": "Decomp_Only", "decomp": True, "recur": False},
        {"name": "Recur_Only", "decomp": False, "recur": True},
        {"name": "Combined", "decomp": True, "recur": True}
    ]

    all_results_dfs = []

    # Sędziowie LLM (inicjowani raz poza pętlą konfiguracji)
    judge_llm = ChatOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct", 
        temperature=0.0,
        max_retries=10,
        request_timeout=300 
    )
    judge_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    my_run_config = RunConfig(max_workers=1, max_retries=10, timeout=300)

    try:
        for config in configs:
            print(f"\n" + "="*50)
            print(f"🚀 ROZPOCZYNAM TEST KONFIGURACJI: {config['name']}")
            print(f"   Dekompozycja: {config['decomp']} | Rekurencja: {config['recur']}")
            print("="*50 + "\n")

            eval_data = {
                "question": [], "answer": [], "contexts": [], "reference": [], "latency": []
            }

            for idx, row in df_original.iterrows():
                question = str(row.get("message", ""))
                original_expert_answer = str(row.get("answer", "")) 

                if pd.isna(row.get("message")) or not question.strip():
                    continue
                    
                print(f"\n[{idx+1}/{len(df_original)}] Pytanie: {question[:80]}...")
                
                try:
                    answer, contexts, latency = run_medical_rag(
                        question, 
                        use_decomposition=config['decomp'], 
                        use_recursion=config['recur']
                    ) 
                    
                    clean_answer = answer.split("DISCLAIMER:")[0].strip()
                    clean_answer = re.sub(r'\s*\[Chunk[^\]]*\]', '', clean_answer)
                    pattern_strict = r"Context:|\n\nContent:"
                    clean_answer = re.sub(pattern_strict, "", clean_answer)
                    
                    if "insufficient clinical data" in answer.lower() or "⚠️" in answer:
                        print("  ⏭️ Bot bezpiecznie odmówił odpowiedzi. Pomijam ewaluację.")
                        continue

                    eval_data["question"].append(question)
                    eval_data["answer"].append(clean_answer)
                    eval_data["contexts"].append(contexts)
                    eval_data["reference"].append(original_expert_answer) 
                    eval_data["latency"].append(latency)
                    
                except Exception as e:
                    print(f"  ❌ Błąd: {e}")

            if len(eval_data['question']) == 0:
                print(f"⚠️ Brak danych do ewaluacji dla konfiguracji {config['name']}. Pomijam.")
                continue

            # Konwersja i Ewaluacja dla danej konfiguracji
            dataset = Dataset.from_dict(eval_data)
            print(f"\n⚖️ Ewaluacja RAGAS dla: {config['name']}...")
            
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, ContextRelevance()],
                llm=judge_llm,
                embeddings=judge_embeddings,
                run_config=my_run_config
            )
            
            df_results = result.to_pandas()
            df_results['config_name'] = config['name'] # Oznaczamy wyniki
            df_results['latency'] = eval_data['latency'] # Przypisujemy czas do tabeli wyników
            all_results_dfs.append(df_results)

            print(f"✅ Podsumowanie dla {config['name']}:")
            print(df_results[['faithfulness', 'answer_relevancy', 'context_precision', 'nv_context_relevance', 'latency']].mean())

        # Zapis zbiorczych wyników
        if all_results_dfs:
            final_df = pd.concat(all_results_dfs, ignore_index=True)
            final_df.to_csv(RESULTS_FILE, index=False)
            print(f"\n💾 ZAPISANO ZBIORCZE WYNIKI DO: {RESULTS_FILE}")
            
            # Generowanie prostej tabeli przestawnej (Summary)
            summary = final_df.groupby('config_name')[['faithfulness', 'answer_relevancy', 'context_precision', 'nv_context_relevance', 'latency']].mean()
            print("\n📊 --- ZBIORCZA ANALIZA PORÓWNAWCZA ---")
            print(summary)

    finally:
        if hasattr(retriever_engine, 'q_client'):
            retriever_engine.q_client.close()

if __name__ == "__main__":
    main()
