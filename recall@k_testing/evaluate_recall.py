import json
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Zmień ścieżkę importu na właściwą dla Twojego projektu
from retriever.retriever import MedicalRetriever

# --- KONFIGURACJA ---
# Lista Twoich dwóch plików z pytaniami
DATASET_FILES = [
    "recall@k_testing/raw_questions.json"#, "recall@k_testing/recall_evaluation_recursive_raw.json" ,
]
COLLECTION = "semantic_raw_v1"
K_VALUES = [1, 3, 5]
OUTPUT_DIR = "recall@k_testing/results"

def get_chunk_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def main():
    print("⚖️ Inicjalizacja zaawansowanej ewaluacji Recall@K...")
    
    # Utworzenie folderu na wyniki, jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Wczytanie i łączenie datasetów
    test_dataset = []
    for file_path in DATASET_FILES:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                test_dataset.extend(data) # Dodajemy pytania do wspólnej puli
                print(f"✅ Wczytano {len(data)} pytań z pliku: {file_path}")
        except FileNotFoundError:
            print(f"⚠️ Nie znaleziono pliku: {file_path}. Pomijam.")

    if not test_dataset:
        print("❌ Brak danych do testowania. Sprawdź ścieżki do plików.")
        return

    # 2. Inicjalizacja Retrievera
    retriever = MedicalRetriever(collection_name=COLLECTION) 
    total_questions = len(test_dataset)
    
    # Lista do zbierania surowych wyników
    evaluation_records = []

    print(f"\n🚀 Rozpoczynam testowanie {total_questions} zapytań (Każde sprawdzane z i bez UMLS)...\n")

    for idx, item in enumerate(test_dataset):
        question = item["question"]
        expected_id = item["expected_chunk_id"]
        # Pobieramy oryginalną nazwę pliku, na bazie którego wygenerowano pytanie
        source_file = os.path.basename(item.get("source_file", "unknown_source"))
        source_label = os.path.splitext(source_file)[0]
        
        print(f"\n[{idx+1}/{total_questions}] Pytanie ({source_file}): {question[:50]}...")
        
        # --- PRZEŁĄCZNIK KOLEKCJI QDRANT W LOCIE ---
        # Dostosuj nazwy kolekcji, jeśli w Qdrant nazywają się inaczej!
        if "recursive" in source_file.lower():
            retriever.collection = "recursive_raw" 
        elif "semantic_raw_v2" in source_file.lower():
            # Ponieważ pytań dostarcza 'semantic_raw.json', upewnij się, czy w bazie nie masz kolekcji 'semantic_raw'
            retriever.collection = "semantic_raw_v2" 
        else:
            retriever.collection = "semantic_raw_v1" 
            
        max_k = max(K_VALUES)
        
        # --- TEST 1: BEZ ONTOLOGII (Baseline) ---
        chunks_no_onto = []
        try:
            chunks_no_onto = retriever.retrieve(user_query=question, use_ontology=False, top_k=max_k)
        except AttributeError as e:
            print(f"   ⚠️ Błąd architektoniczny (brak metody fallback?): {e}. Zapisuję jako 0 wyników.")
        except Exception as e:
            print(f"   ⚠️ Nieznany błąd podczas pobierania: {e}. Zapisuję jako 0 wyników.")

        ids_no_onto = [get_chunk_id(chunk["text"]) for chunk in chunks_no_onto]
        
        record_no_onto = {
            "Question ID": idx + 1,
            "Source File": source_label,
            "Chunking Type": "Recursive" if "recursive" in source_file else "Semantic",
            "UMLS (Ontology)": "False (Baseline)"
        }
        for k in K_VALUES:
            record_no_onto[f"Hit@{k}"] = 1 if expected_id in ids_no_onto[:k] else 0
        evaluation_records.append(record_no_onto)
        
        # # --- TEST 2: Z ONTOLOGIĄ (Boosted) ---
        # chunks_with_onto = []
        # try:
        #     chunks_with_onto = retriever.retrieve(user_query=question, use_ontology=True, top_k=max_k)
        # except AttributeError as e:
        #     print(f"   ⚠️ Błąd architektoniczny (brak metody fallback?): {e}. Zapisuję jako 0 wyników.")
        # except Exception as e:
        #     print(f"   ⚠️ Nieznany błąd podczas pobierania: {e}. Zapisuję jako 0 wyników.")

        # ids_with_onto = [get_chunk_id(chunk["text"]) for chunk in chunks_with_onto]
        
        # record_with_onto = {
        #     "Question ID": idx + 1,
        #     "Chunking Type": "Recursive" if "recursive" in source_file else "Semantic"
        # }
        # for k in K_VALUES:
        #     record_with_onto[f"Hit@{k}"] = 1 if expected_id in ids_with_onto[:k] else 0
        # evaluation_records.append(record_with_onto)

    # Bezpieczne zamknięcie klienta Qdrant
    if hasattr(retriever, 'q_client'):
        retriever.q_client.close()

    # ==========================================
    # 3. AGREGACJA DANYCH I TABELE (PANDAS)
    # ==========================================
    df = pd.DataFrame(evaluation_records)
    
    # Obliczanie średniej dla każdego wariantu (* 100 aby mieć procenty)
    # Grupuj po oryginalnym pliku źródłowym, aby porównać wszystkie trzy zestawy pytań osobno
    summary_df = df.groupby(["Source File", "UMLS (Ontology)"])[[f"Hit@{k}" for k in K_VALUES]].mean() * 100
    summary_df = summary_df.round(2) 
    
    # Zmiana nazw kolumn
    summary_df = summary_df.rename(columns={f"Hit@{k}": f"Recall@{k} [%]" for k in K_VALUES})
    
    print("\n" + "="*70)
    print("📊 ZBIORCZY RAPORT EWALUACYJNY: RECALL@K")
    print("="*70)
    print(summary_df.to_markdown())
    print("="*70)

    # Zapis wyników do plików CSV
    raw_csv_path = os.path.join(OUTPUT_DIR, "detailed_recall_results_raw.csv")
    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_recall_results_raw.csv")
    df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path)
    print(f"💾 Zapisano szczegółowe wyniki do: {raw_csv_path}")
    print(f"💾 Zapisano tabelę podsumowującą do: {summary_csv_path}")

    # ==========================================
    # 4. GENEROWANIE WYKRESÓW (SEABORN/MATPLOTLIB)
    # ==========================================
    print("\n📈 Generowanie wykresów...")
    
    df_melted = df.melt(id_vars=["Question ID", "Source File", "Chunking Type", "UMLS (Ontology)"], 
                        value_vars=[f"Hit@{k}" for k in K_VALUES],
                        var_name="Metric", 
                        value_name="Score")
    df_melted["Score"] = df_melted["Score"] * 100 

    sns.set_theme(style="whitegrid")
    
    g = sns.catplot(
        data=df_melted, kind="bar",
        x="Metric", y="Score", hue="UMLS (Ontology)", col="Source File",
        errorbar=None, palette="dark:salmon_r", alpha=.6, height=6, aspect=1.2
    )
    
    g.despine(left=True)
    g.set_axis_labels("", "Recall (%)")
    g.legend.set_title("UMLS Support")
    
    # Etykiety z procentami nad słupkami
    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.1f}%", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=10, fontweight='bold')
        ax.set_ylim(0, 110)

    plt.subplots_adjust(top=0.85)
    g.figure.suptitle('Recall@K Comparison', fontsize=16, fontweight='bold')

    plot_path = os.path.join(OUTPUT_DIR, "recall_comparison_chart_raw.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Wykres zapisany jako: {plot_path}")
    
if __name__ == "__main__":
    main()