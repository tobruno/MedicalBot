import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# KONFIGURACJA
# ==========================================
BASE_FILE = 'ragas/testing/liveqa_sampled_evaluated.csv'
SAMPLE_SIZE = 50  # Liczba pytań rzeczywiście brana pod uwagę w ewaluacji

# Słownik z wynikami ewaluacji: {'Nazwa Metody': 'sciezka_do_pliku.csv'}
RESULTS_FILES = {
    'Recursive Contextualised': 'ragas/testing/ragas_liveqa_results_recursive.csv',
    'Semantic Contextualised': 'ragas/testing/ragas_liveqa_results_semantic.csv'
    # Możesz tutaj dodawać kolejne metody, np.:
    #'Semantic': 'ragas/testing/ragas_medredqa_results_semantic_raw.csv'
}

METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'nv_context_relevance']

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def clean_text(text):
    """Standaryzuje tekst, aby ułatwić łączenie tabel (usuwa znaki specjalne, podwójne spacje)."""
    if not isinstance(text, str): return ""
    text = re.sub(r'\W+', ' ', text)
    return text.lower().strip()

# ==========================================
# GŁÓWNA ANALIZA
# ==========================================
def run_analysis():
    # 1. Wczytanie pliku bazowego i ograniczenie do analizowanej próbki
    print("Wczytywanie pliku bazowego...")
    base_df = pd.read_csv(BASE_FILE)
    base_df = base_df.head(SAMPLE_SIZE).copy()  # Bierzemy tylko tyle pytań, ile faktycznie testowano
    base_df['Match_Key'] = base_df['message'].apply(clean_text)
    
    analysis_results = {}
    
    # 2. Przetwarzanie wyników każdej metody
    for method_name, file_path in RESULTS_FILES.items():
        print(f"Przetwarzanie wyników dla metody: {method_name}...")
        res_df = pd.read_csv(file_path)
        res_df['Match_Key'] = res_df['user_input'].apply(clean_text)
        
        # Łączenie z plikiem bazowym (Left join do pełnej próbki 50 pytań)
        merged_df = pd.merge(base_df, res_df, on='Match_Key', how='left')
        
        # Oznaczenie sukcesu (czy udało się ocenić faithfulness)
        merged_df['Success'] = merged_df['faithfulness'].notna()
        analysis_results[method_name] = merged_df

    # 3. Obliczenia statystyk
    print("\n--- PODSUMOWANIE WYNIKÓW OGÓLNYCH ---")
    
    summary_data = []
    success_rates = {}
    success_counts = {}

    answered_keys_by_method = {
        method: set(df.loc[df['Success'], 'Match_Key'])
        for method, df in analysis_results.items()
    }
    common_answered_keys = set.intersection(*answered_keys_by_method.values()) if answered_keys_by_method else set()
    common_count = len(common_answered_keys)
    print(f"\nWspólna liczba pytań ocenionych przez wszystkie metody: {common_count}/{SAMPLE_SIZE}")
    if common_count == 0:
        raise ValueError("Brak wspólnie ocenionych pytań. Sprawdź pliki wynikowe.")

    for method, df in analysis_results.items():
        # Success Rate
        success_count = df['Success'].sum()
        success_rate = (success_count / SAMPLE_SIZE) * 100
        success_rates[method] = success_rate
        success_counts[method] = success_count
        print(f"[{method}] Odpowiedzi: {success_count}/{SAMPLE_SIZE} ({success_rate:.1f}%)")
        
        # Średnie metryki tylko dla wspólnie ocenionych pytań
        common_df = df[df['Match_Key'].isin(common_answered_keys)]
        means = common_df[METRICS].mean().to_dict()
        means['Metoda'] = method
        summary_data.append(means)
        
    summary_df = pd.DataFrame(summary_data).set_index('Metoda')
    print("\nŚrednie wartości metryk RAGAS:")
    print(summary_df.round(3))
    # Zapis zbiorczych wyników do CSV
    summary_df.to_csv('ragas/results/ragas_automated_summary_overall_liveqa.csv')
    
    # ==========================================
    # WIZUALIZACJE
    # ==========================================
    
    # Wykres 1: Porównanie metryk

    ax = summary_df.T.plot(kind='bar', figsize=(10, 6), width=0.6)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    plt.title('Ogólne porównanie metryk RAGAS')
    plt.ylabel('Średni wynik')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metoda')
    plt.tight_layout()
    plt.savefig('ragas/results/porownanie_metryk_ogolne_liveqa.png')

    
    # Wykres 2: Wskaźnik sukcesu (Success Rate ogólny)
    plt.figure(figsize=(8, 6))
    # Zwężone słupki na wykresie dla lepszej estetyki przy mniejszej ich ilości
    bars = plt.bar(success_rates.keys(), success_rates.values(), color=['skyblue', 'lightgreen', 'salmon'], width=0.5)
    plt.title(f'Ogólny wskaźnik sukcesu (próba = {SAMPLE_SIZE} pytań)')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105)
    
    # Dodanie wartości procentowych i ułamków nad słupkami
    for bar, method in zip(bars, success_rates.keys()):
        yval = bar.get_height()
        count = success_counts[method]
        label_text = f"{yval:.1f}%\n({count}/{SAMPLE_SIZE})"
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, label_text, ha='center', va='bottom', fontweight='bold')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('ragas/results/wskaznik_sukcesu_ogolny_liveqa.png')

    print("\nAnaliza zakończona. Wygenerowano 2 pliki PNG i 1 plik CSV w folderze 'ragas/'.")

if __name__ == "__main__":
    run_analysis()