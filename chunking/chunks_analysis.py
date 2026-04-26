import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_extract_lengths(file_path, label):
    """Wczytuje plik JSON i wyciąga długość każdego chunka w znakach i słowach."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Nie znaleziono pliku: {file_path}")
        return pd.DataFrame()
    
    lengths_chars = []
    lengths_words = []
    
    for item in data:
        text = item.get("text", "")
        lengths_chars.append(len(text))
        lengths_words.append(len(text.split()))
        
    return pd.DataFrame({
        'Metoda': label,
        'Znaki': lengths_chars,
        'Słowa': lengths_words
    })

def analyze_and_compare(file1, file2, label1="Wariant 1", label2="Wariant 2"):
    print("⏳ Trwa wczytywanie i analiza danych...")
    
    # 1. Ładowanie danych
    df1 = load_and_extract_lengths(file1, label1)
    df2 = load_and_extract_lengths(file2, label2)
    
    if df1.empty or df2.empty:
        print("Brak danych do porównania. Sprawdź ścieżki plików.")
        return
    
    # Łączymy dane w jedną tabelę
    df = pd.concat([df1, df2], ignore_index=True)
    
    # 2. Obliczanie statystyk opisowych
    print(f"\n{'='*50}")
    print(f"📊 ANALIZA PORÓWNAWCZA: {label1} vs {label2}")
    print(f"{'='*50}")
    
    # Statystyki dla Znaków
    stats_chars = df.groupby('Metoda')['Znaki'].agg(
        Ilość_Chunków='count',
        Średnia='mean',
        Mediana='median',
        Min='min',
        Max='max',
        Odchylenie_Std='std'
    ).round(1)
    
    print("\n📈 [STATYSTYKI - Długość w ZNAKACH]:")
    print(stats_chars.to_string())

    # Statystyki dla Słów
    stats_words = df.groupby('Metoda')['Słowa'].agg(
        Ilość_Chunków='count',
        Średnia='mean',
        Mediana='median',
        Min='min',
        Max='max',
        Odchylenie_Std='std'
    ).round(1)
    
    print("\n📈 [STATYSTYKI - Długość w SŁOWACH]:")
    print(stats_words.to_string())

    # 3. Generowanie wykresów
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Wykres A: Histogram + Linia gęstości (KDE)
    sns.histplot(data=df, x='Znaki', hue='Metoda', kde=True, ax=axes[0], bins=40, alpha=0.5)
    axes[0].set_title('A: Rozkład długości chunków (Znaki)', fontsize=14)
    axes[0].set_xlabel('Liczba znaków')
    axes[0].set_ylabel('Liczba chunków')

    # Wykres B: Wykres pudełkowy (Boxplot) - świetny do pokazywania rozrzutu i anomalii
    sns.boxplot(data=df, x='Metoda', y='Znaki', hue='Metoda', ax=axes[1], palette="Set2", legend=False)
    axes[1].set_title('B: Rozrzut długości i wartości skrajne', fontsize=14)
    axes[1].set_xlabel('Metoda')
    axes[1].set_ylabel('Liczba znaków')

    plt.tight_layout()
    plot_filename = "chunks/porownanie_chunkow.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\n🖼️ Wygenerowano wizualizację i zapisano jako: '{plot_filename}'")
    
    # Wyświetlenie wykresu na ekranie
    plt.show()

if __name__ == "__main__":
    # --- TUTAJ PODAJ SWOJE PLIKI ---
    PLIK_1 = "chunks/chunks_semantic_raw_2_95.json"
    PLIK_2 = "chunks/chunks_semantic_raw.json"
    
    # Możesz zmienić nazwy na ekranie (etykiety) zależnie od tego co testujesz
    analyze_and_compare(
        file1=PLIK_1, 
        file2=PLIK_2, 
        label1="Recursive Chunking", 
        label2="Semantic Chunking"
    )