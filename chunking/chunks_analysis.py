# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_and_extract_lengths(file_path, label):
#     """Wczytuje plik JSON i wyciąga długość każdego chunka w znakach i słowach."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#     except FileNotFoundError:
#         print(f"❌ Nie znaleziono pliku: {file_path}")
#         return pd.DataFrame()
    
#     lengths_chars = []
#     lengths_words = []
    
#     for item in data:
#         text = item.get("text", "")
#         lengths_chars.append(len(text))
#         lengths_words.append(len(text.split()))
        
#     return pd.DataFrame({
#         'Metoda': label,
#         'Znaki': lengths_chars,
#         'Słowa': lengths_words
#     })

# def analyze_and_compare(file1, file2, label1="Wariant 1", label2="Wariant 2"):
#     print("⏳ Trwa wczytywanie i analiza danych...")
    
#     # 1. Ładowanie danych
#     df1 = load_and_extract_lengths(file1, label1)
#     df2 = load_and_extract_lengths(file2, label2)
    
#     if df1.empty or df2.empty:
#         print("Brak danych do porównania. Sprawdź ścieżki plików.")
#         return
    
#     # Łączymy dane w jedną tabelę
#     df = pd.concat([df1, df2], ignore_index=True)
    
#     # 2. Obliczanie statystyk opisowych
#     print(f"\n{'='*50}")
#     print(f"📊 ANALIZA PORÓWNAWCZA: {label1} vs {label2}")
#     print(f"{'='*50}")
    
#     # Statystyki dla Znaków
#     stats_chars = df.groupby('Metoda')['Znaki'].agg(
#         Ilość_Chunków='count',
#         Średnia='mean',
#         Mediana='median',
#         Min='min',
#         Max='max',
#         Odchylenie_Std='std'
#     ).round(1)
    
#     print("\n📈 [STATYSTYKI - Długość w ZNAKACH]:")
#     print(stats_chars.to_string())

#     # Statystyki dla Słów
#     stats_words = df.groupby('Metoda')['Słowa'].agg(
#         Ilość_Chunków='count',
#         Średnia='mean',
#         Mediana='median',
#         Min='min',
#         Max='max',
#         Odchylenie_Std='std'
#     ).round(1)
    
#     print("\n📈 [STATYSTYKI - Długość w SŁOWACH]:")
#     print(stats_words.to_string())

#     # 3. Generowanie wykresów
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Wykres A: Histogram + Linia gęstości (KDE)
#     sns.histplot(data=df, x='Znaki', hue='Metoda', kde=True, ax=axes[0], bins=40, alpha=0.5)
#     axes[0].set_title('A: Rozkład długości chunków (Znaki)', fontsize=14)
#     axes[0].set_xlabel('Liczba znaków')
#     axes[0].set_ylabel('Liczba chunków')

#     # Wykres B: Wykres pudełkowy (Boxplot) - świetny do pokazywania rozrzutu i anomalii
#     sns.boxplot(data=df, x='Metoda', y='Znaki', hue='Metoda', ax=axes[1], palette="Set2", legend=False)
#     axes[1].set_title('B: Rozrzut długości i wartości skrajne', fontsize=14)
#     axes[1].set_xlabel('Metoda')
#     axes[1].set_ylabel('Liczba znaków')

#     plt.tight_layout()
#     plot_filename = "chunking/porownanie_chunkow.png"
#     plt.savefig(plot_filename, dpi=300)
#     print(f"\n🖼️ Wygenerowano wizualizację i zapisano jako: '{plot_filename}'")
    
#     # Wyświetlenie wykresu na ekranie
#     plt.show()

# if __name__ == "__main__":
#     # --- TUTAJ PODAJ SWOJE PLIKI ---
#     PLIK_1 = "chunking/chunks_recursive_raw.json"
#     PLIK_2 = "chunking/chunks_semantic_raw.json"
    
#     # Możesz zmienić nazwy na ekranie (etykiety) zależnie od tego co testujesz
#     analyze_and_compare(
#         file1=PLIK_1, 
#         file2=PLIK_2, 
#         label1="Recursive Chunking", 
#         label2="Semantic Chunking"
#     )
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_extract_features(file_path, label):
    """Wczytuje dane i wyciąga zaawansowane cechy tekstowe."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku: {file_path}")
        return pd.DataFrame()
    
    features = []
    for item in data:
        text = item.get("text", "")
        words = text.split()
        char_count = len(text)
        word_count = len(words)
        # Estymacja tokenów dla systemów RAG (średnio 1.3 tokena na słowo)
        token_est = int(word_count * 1.3) 
        
        features.append({
            'Metoda': label,
            'Znaki': char_count,
            'Słowa': word_count,
            'Tokeny_Est': token_est
        })
        
    return pd.DataFrame(features)

def analyze_for_thesis(file1, file2, label1="Recursive", label2="Semantic"):
    # 1. Przygotowanie danych
    df1 = load_and_extract_features(file1, label1)
    df2 = load_and_extract_features(file2, label2)
    df = pd.concat([df1, df2], ignore_index=True)

    # --- Compatibility aliases ---
    # Some plotting code expects English-named columns. Create aliases
    # so both Polish and English names are available.
    if 'Znaki' in df.columns:
        df['Characters'] = df['Znaki']
    if 'Słowa' in df.columns:
        df['Words'] = df['Słowa']
    if 'Tokeny_Est' in df.columns:
        df['Tokens_Est'] = df['Tokeny_Est']
    if 'Metoda' in df.columns:
        df['Method'] = df['Metoda']

    # 2. Statystyki opisowe
    stats_summary = df.groupby('Metoda').agg({
        'Znaki': ['mean', 'std', 'max'],
        'Słowa': ['mean', 'std', 'max'],
        'Tokeny_Est': ['mean', 'std']
    }).round(1)
    
    # Obliczanie Współczynnika Zmienności (Coefficient of Variation) CV = (std/mean)*100
    print("\n📊 TABELA STATYSTYCZNA DO PRACY MAGISTERSKIEJ:")
    print(stats_summary.to_string())

    # 3. WIZUALIZACJA - Układ 2x2
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A. Histogram ZNAKÓW
    sns.histplot(data=df, x='Characters', hue='Method', kde=True, ax=axes[0, 0], element="step")
    axes[0, 0].set_title('A: Character Length Distribution', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Number of Characters')

    # B. Histogram SŁÓW (ZAMIANA ECDF)
    sns.histplot(data=df, x='Słowa', hue='Metoda', kde=True, ax=axes[0, 1], element="step", palette="viridis")
    axes[0, 1].set_title('B: Word Length Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Words')

    # C. Wykres skrzypcowy TOKENÓW (Estymacja kosztowa/techniczna)
    sns.violinplot(data=df, x='Metoda', y='Tokeny_Est', inner="quart", ax=axes[1, 0])
    axes[1, 0].set_title('C: Density and Quartiles of Tokens', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('Estimated Tokens (Cost Proxy)')

    # D. Wykres pudełkowy SŁÓW (Outliery i spójność)
    sns.boxplot(data=df, x='Metoda', y='Słowa', ax=axes[1, 1])
    axes[1, 1].set_title('D: Distribution of Words (Boxplot)', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Words')

    plt.suptitle(f"Comparison of Chunking Methods for MedicalBot", fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_name = "chunking/medicalbot_full_analysis.png"
    #plt.savefig(output_name, dpi=300)
    print(f"\n🖼️ Wykres zapisany jako: {output_name}")
    plt.show()

if __name__ == "__main__":
    # Ścieżki do Twoich plików
    #analyze_for_thesis("chunking/chunks_recursive_raw.json", "chunking/chunks_semantic_raw.json")
    analyze_for_thesis("chunking/chunks_semantic_raw_v2.json", "chunking/chunks_semantic_raw_88.json")