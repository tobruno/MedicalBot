import pandas as pd
import re
from datasets import load_dataset

# --- KONFIGURACJA ---
DATASET_NAME = "truehealth/liveqa"
INPUT_FILE = "medredqa/medredqa_train.csv"

def calculate_text_stats(series, name, original_length):
    """Oblicza statystyki długości i złożoności dla podanej kolumny tekstowej."""
    series = series.dropna().astype(str)
    
    # Podstawowe miary
    word_counts = series.apply(lambda x: len(x.split()))
    char_counts = series.str.len()
    
    # Przybliżona liczba zdań
    sentence_counts = series.apply(lambda x: len([s for s in re.split(r'[.!?]+', x) if s.strip()])).replace(0, 1)
    
    # ZŁOŻONOŚĆ: Liczenie znaków zapytania (ile faktycznie pytań pada w jednym tekście)
    questions_per_post = series.apply(lambda x: x.count('?'))
    
    # Metryki złożoności
    avg_word_length = char_counts.sum() / word_counts.sum() if word_counts.sum() > 0 else 0
    words_per_sentence = word_counts / sentence_counts
    
    print(f"\n📊 Statystyki dla zbioru: {name}")
    print("-" * 50)
    print(f"Początkowa pula pytań      : {original_length}")
    print(f"Pytań po czyszczeniu (baza): {len(series)} (odrzucono {original_length - len(series)})")
    print(f"Średnia liczba słów        : {word_counts.mean():.1f} (Max: {word_counts.max()})")
    print(f"Śr. długość słowa (znaki)  : {avg_word_length:.2f} (wyższa = trudniejsze słownictwo)")
    print(f"Śr. słów na zdanie         : {words_per_sentence.mean():.1f} (wyższa = bardziej złożone zdania)")
    print(f"Śr. znaków '?' w tekście   : {questions_per_post.mean():.2f} (wyższa = wielowątkowość zapytania)")
    print("-" * 50)
    
    return {
        'Dataset': name,
        'Oryginalna Ilość': original_length,
        'Ilość po filtrach': len(series),
        'Śr. Słów': round(word_counts.mean(), 1),
        'Śr. Pytań w poście (?)': round(questions_per_post.mean(), 2),
        'Śr. Słów/Zdanie': round(words_per_sentence.mean(), 1)
    }

def analyze_medredqa_sample():
    print(f"📥 Pobieranie datasetu '{INPUT_FILE}'...")
    try:
        dataset = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Błąd: Nie znaleziono pliku {INPUT_FILE}. Upewnij się, że ścieżka jest poprawna.")
        return None

    original_length = len(dataset)
    
    print("🧹 Czyszczenie MedRedQA (usuwanie pustych, filtrowanie wyników i słów kluczowych)...")
    dataset = dataset[dataset['Occupation'] != 'Medical Student']
    dataset = dataset.dropna(subset=['Body', 'Response'])
    dataset = dataset[dataset['Response Score'] > 20]
    
    banned_words = 'EDIT|UPDATE|CORRECTION|Sex:|Sex M|Sex F|photo|image'
    dataset = dataset[~dataset['Body'].str.contains(banned_words, case=False, na=False)]
    
    stats = calculate_text_stats(dataset['Body'], "MedRedQA", original_length)
    return stats

def analyze_liveqa_sample():
    print(f"\n📥 Pobieranie datasetu '{DATASET_NAME}' z Hugging Face...")
    dataset = load_dataset(DATASET_NAME, split="train")
    df = dataset.to_pandas()
    
    original_length = len(df)
    
    print("🧹 Czyszczenie LiveQA (usuwanie pustych zapytań)...")
    df['message'] = df['message'].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=['message'])
    
    stats = calculate_text_stats(df['message'], "LiveQA", original_length)
    return stats

if __name__ == "__main__":
    print("🚀 Rozpoczynam analizę porównawczą...")
    
    stats_medred = analyze_medredqa_sample()
    stats_liveqa = analyze_liveqa_sample()
    
    if stats_medred and stats_liveqa:
        print("\n🏆 Zestawienie porównawcze (Podsumowanie):")
        comparison_df = pd.DataFrame([stats_medred, stats_liveqa])
        print(comparison_df.to_string(index=False))