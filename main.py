from retriever.decomposer import decompose_query
from retriever.retriever import MedicalRetriever
from generator import generate_grounded_response

# Inicjalizacja Twojego silnika Qdrant
retriever_engine = MedicalRetriever()

def run_medical_rag(user_complex_query):
    # 1. Dekompozycja (Magia Agenta)
    atomic_queries = decompose_query(user_complex_query)
    print(f"📌 Zidentyfikowano intencje ({len(atomic_queries)}): {atomic_queries}")
    
    all_retrieved_chunks = []
    seen_chunk_texts = set() # Zbiór do zapobiegania duplikatom
    
    # 2. Równoległe/Sekwencyjne wyszukiwanie dla każdej intencji
    for idx, atomic_query in enumerate(atomic_queries):
        print(f"🔍 Szukanie w bazie dla intencji {idx+1}: '{atomic_query}'")
        
        # Używamy Twojego istniejącego retrievera!
        # Zmniejszamy top_k do np. 3, żeby nie przeładować kontekstu przy 3 różnych intencjach
        chunks = retriever_engine.retrieve(user_query=atomic_query, use_ontology=True, top_k=3)
        
        # 3. Deduplikacja (jeśli dwa zapytania zwrócą ten sam dokument)
        for chunk in chunks:
            if chunk['text'] not in seen_chunk_texts:
                all_retrieved_chunks.append(chunk)
                seen_chunk_texts.add(chunk['text'])
                
    print(f"📚 Zgromadzono łącznie {len(all_retrieved_chunks)} unikalnych chunków wiedzy.")

    # 4. Finalna Generacja (Generator dostaje oryginalne, złożone pytanie i połączony kontekst)
    if not all_retrieved_chunks:
        all_retrieved_chunks = [{"text": "Lack of medical data."}]
        

    # # =========================================================
    # # 🔍 DODAJ TEN FRAGMENT: WYŚWIETLANIE CHUNKÓW W KONSOLI
    # # =========================================================
    # print("\n--- 📄 PODGLĄD ZNALEZIONYCH DOKUMENTÓW ---")
    # for i, chunk in enumerate(all_retrieved_chunks):
    #     # Wyciągamy tekst, zamieniamy entery na spacje dla czytelności w terminalu
    #     # i ucinamy do pierwszych 300 znaków, żeby podgląd był zwięzły
    #     clean_text = chunk['text'].replace('\n', ' ')
    #     preview = f"{clean_text[:300]}..." if len(clean_text) > 300 else clean_text
        
    #     print(f"[{i+1}] {preview}")
        
    #     # Jeśli chcesz widzieć ewentualne dopasowania ontologiczne z UMLS (opcjonalne):
    #     if 'umls_overlap' in chunk and chunk['umls_overlap']:
    #         print(f"    🧬 Znalezione pojęcia UMLS: {chunk['umls_overlap']}")
    # print("------------------------------------------\n")
    # # =========================================================

    # # 4. Finalna Generacja (Generator dostaje oryginalne, złożone pytanie i połączony kontekst)
    # if not all_retrieved_chunks:
    #     all_retrieved_chunks = [{"text": "Lack of medical data."}]
        
    final_response = generate_grounded_response(user_complex_query, all_retrieved_chunks)
    
    return final_response


# --- TEST ---
if __name__ == "__main__":
    #test_query = "Jakie jest dawkowanie paracetamolu u dzieci i czy witamina C leczy raka?"
    test_query = "What is the recommended dosage of paracetamol for children and does vitamin C cure cancer?"
    answer = run_medical_rag(test_query)
    print("\n🩺 ODPOWIEDŹ BOTA:\n", answer["answer"])
    retriever_engine.q_client.close()