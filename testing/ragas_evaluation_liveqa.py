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
    context_revelancy,
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
QUESTIONS_FILE = "testing/liveqa_sampled_evaluated.csv"
RESULTS_FILE = "testing/ragas_liveqa_results.csv"
SAMPLE_SIZE = 20 # Ustaw limit pytań do testów (generowanie i ocena trwa!). Ustaw na None, aby przetworzyć wszystkie.

print("🔧 Inicjalizacja silnika Qdrant...")
retriever_engine = MedicalRetriever()

def run_medical_rag(user_complex_query):
    """
    Uruchamia Twój pipeline RAG i zwraca wygenerowaną odpowiedź 
    ORAZ listę użytych kontekstów (wymagane dla RAGAS).
    """
    # 1. Dekompozycja
    atomic_queries = decompose_query(user_complex_query)
    print(f"  📌 Zidentyfikowano intencje: {atomic_queries}")
    
    all_retrieved_chunks = []
    seen_chunk_texts = set()
    
    # 2. Wyszukiwanie
    for atomic_query in atomic_queries:
        chunks = retriever_engine.retrieve(user_query=atomic_query, use_ontology=True, top_k=3)
        for chunk in chunks:
            if chunk['text'] not in seen_chunk_texts:
                all_retrieved_chunks.append(chunk)
                seen_chunk_texts.add(chunk['text'])
                
    # 3. Zabezpieczenie przed brakiem danych
    if not all_retrieved_chunks:
        all_retrieved_chunks = [{"text": "Lack of medical data."}]
        
    # 4. Generacja
    # Zakładam, że Twoja funkcja zwraca słownik, z którego odpowiedź wyciągasz pod kluczem ["answer"]
    raw_response = generate_grounded_response(user_complex_query, all_retrieved_chunks)
    
    # Wyciągamy sam tekst odpowiedzi
    if isinstance(raw_response, dict) and "answer" in raw_response:
        final_answer = raw_response["answer"]
    else:
        final_answer = str(raw_response) # Fallback, jeśli zwraca stringa

    # Zbieramy same teksty chunków dla ewaluatora RAGAS
    contexts_texts = [chunk['text'] for chunk in all_retrieved_chunks]

    return final_answer, contexts_texts

def main():
    print(f"📊 Wczytywanie pytań z {QUESTIONS_FILE}...")
    df = pd.read_csv(QUESTIONS_FILE)
    
    if SAMPLE_SIZE:
        df = df.head(SAMPLE_SIZE)
        print(f"⚠️ Uwaga: Przetwarzam tylko pierwsze {SAMPLE_SIZE} pytań w celach testowych.")

    # --- NOWE: Dodano klucz "reference" ---
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "reference": [] 
    }
    
    print("\n⚙️ Rozpoczynam dynamiczne generowanie odpowiedzi...")
    for idx, row in df.iterrows():
        question = str(row.get("message", ""))
        
        # --- NOWE: Pobieramy oryginalną odpowiedź eksperta z CSV ---
        original_expert_answer = str(row.get("answer", "")) 

        if pd.isna(row.get("message")) or not question.strip():
            continue
            
        print(f"\n[{idx+1}/{len(df)}] Pytanie: {question[:100]}...")
        
        try:
            answer, contexts = run_medical_rag(question) 
            clean_answer = answer.split("Disclaimer:")[0].strip()
            clean_answer = re.sub(r'\s*\[Chunk[^\]]*\]', '', clean_answer)
            
            if "insufficient clinical data" in answer.lower() or "⚠️" in answer:
                print("  ⏭️ Bot bezpiecznie odmówił odpowiedzi. Pomijam ewaluację.")
                continue

            eval_data["question"].append(question)
            eval_data["answer"].append(clean_answer)  # Odpowiedź z Twojego RAGa
            eval_data["contexts"].append(contexts)
            # --- NOWE: Dodajemy odpowiedź z CSV jako punkt odniesienia ---
            eval_data["reference"].append(original_expert_answer) 
            
        except Exception as e:
            print(f"  ❌ Błąd podczas generowania odpowiedzi: {e}")

    if len(eval_data['question']) == 0:
        print("Brak poprawnie wygenerowanych odpowiedzi do ewaluacji. Kończę.")
        return

    # Zamiana na format HuggingFace Dataset
    dataset = Dataset.from_dict(eval_data)

    print("\n⚖️ Inicjalizacja sędziów LLM (OpenRouter & Ollama)...")
    # judge_llm = ChatGroq(
    #     api_key=os.environ.get("GROQ_API_KEY"),
    #     model_name="llama-3.3-70b-versatile",  #"llama-3.3-70b-versatile",
    #     temperature=0.0,
    #     max_retries=10, # LangChain będzie czekał i próbował ponownie
    #     request_timeout=300
    # )
    judge_llm = ChatOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        # Używamy Gemini, bo jest potężne, nie ma restrykcji z parametrem 'n' i ma ogromne limity
        model="meta-llama/llama-3.3-70b-instruct", 
        temperature=0.0,
        max_retries=10,
        request_timeout=300 
    )
    
    # --- NOWE: Zaktualizowany import Ollamy (po instalacji langchain-ollama) ---
    judge_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("\n🚀 Uruchamiam ewaluację RAGAS...")
    try:
        my_run_config = RunConfig(max_workers=1, max_retries=10, timeout=300)
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,      
                answer_relevancy,  
                context_precision,
                context_revelancy  # Teraz zadziała, bo dodaliśmy 'reference'
            ],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=my_run_config
        )
        
        df_results = result.to_pandas()
        print("\n✅ Ewaluacja Zakończona!")
        print("\n--- ZBIORCZE WYNIKI ---")
        print(result)
        
        df_results.to_csv(RESULTS_FILE, index=False)
        print(f"\n💾 Szczegółowy raport zapisano do: {RESULTS_FILE}")
        print("Średnie: ", df_results.mean(numeric_only=True))  # Podsumowanie średnich wyników dla każdej metryki
        #print(f"Średnia faithfulness: {df_results.faithfulness.mean()}, answer_relevancy: {df_results.answer_relevancy.mean()}, context_precision: {df_results.context_precision.mean()}")  # Podgląd szczegółowych wyników per pytanie

        
    finally:
        # Zabezpieczenie przed dziwnymi błędami Qdrant (jak "sys.meta_path is None")
        # Ensure client is closed regardless of whether evaluation fails or succeeds.
        if hasattr(retriever_engine, 'q_client'):
            retriever_engine.q_client.close()

if __name__ == "__main__":
    main()
