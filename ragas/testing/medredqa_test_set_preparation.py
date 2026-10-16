import pandas as pd
import ollama
import re
import os
from datasets import load_dataset

# --- KONFIGURACJA ---
INPUT_FILE = "medredqa/medredqa_train.csv"
SAMPLE_SIZE = 100  # Ile pytań chcesz wylosować do testów
OUTPUT_FILE = "ragas/testing/medredqa_sampled_evaluated.csv"
TOPICS_FILE = "encyclopedia/clean_list_of_entries.txt"

def load_encyclopedia_topics():
    """Wczytuje listę tematów z pliku tekstowego jako pojedynczy string."""
    if not os.path.exists(TOPICS_FILE):
        raise FileNotFoundError(f"❌ Nie znaleziono pliku {TOPICS_FILE}! Upewnij się, że jest w tym samym folderze.")
    
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        # Odczytujemy całość i czyścimy ewentualne białe znaki
        return f.read().strip()

def download_and_sample_dataset():
    """Pobiera dataset z Hugging Face, usuwa puste wiersze w 'message' i zwraca próbkę jako DataFrame."""
    
    print(f"📥 Pobieranie datasetu '{INPUT_FILE}'...")
    dataset = pd.read_csv(INPUT_FILE)
    dataset = dataset[dataset['Occupation'] != 'Medical Student']
    # Wyrzucamy posty dłuższe niż 300 słów
    dataset = dataset[dataset['Body'].apply(lambda x: len(str(x).split()) <= 250)]
    dataset = dataset[dataset['Response Score']>20]
    dataset = dataset.loc[dataset['Body'].str.contains('EDIT|UPDATE|CORRECTION|Sex:|Sex M|Sex F|photo|image|post|reddit|upvote|lawyer|sue|insurance|bill|cost|pro bono|EMTALA', case=False, na=False) == False]
    # Usuwanie wierszy, gdzie 'message' jest puste, None lub składa się tylko ze spacji
    print("🧹 Usuwanie pustych wierszy w kolumnie 'Body'...")
    dataset = dataset.dropna(subset=['Body'])
    dataset = dataset.dropna(subset=['Response'])

    print(f"📊 Liczba pytań po filtracji: {len(dataset)}")
    
    print(f"🔀 Losowanie próbki {SAMPLE_SIZE} pytań (seed=42)...")
    # Upewniamy się, że nie próbujemy pobrać więcej wierszy niż zostało po filtracji
    #actual_sample_size = min(SAMPLE_SIZE, len(dataset))
    sampled_dataset = dataset.sample(n=min(SAMPLE_SIZE, len(dataset)), random_state=42)
    #sampled_dataset = dataset.shuffle(seed=42).select(range(actual_sample_size))
    
    return sampled_dataset

def assess_question_with_llm(question, encyclopedia_topics):
    """
    Używa lokalnej Llamy (przez Ollamę) do oceny Trudności i Trafności.
    Działa w 100% offline, brak limitów zapytań (Rate Limits).
    """
    prompt = f"""
        <role>
        You are an expert Clinical Triage Assessor and Data Evaluator. 
        Your task is to evaluate a patient's medical question on two dimensions: Difficulty and Relevance to a specific medical encyclopedia.
        </role>

        <encyclopedia_index>
        {encyclopedia_topics}
        </encyclopedia_index>

        <question>
        {question}
        </question>

        <instructions>
        TASK 1: DIFFICULTY LEVEL
        Classify the question's complexity based on the required medical knowledge:
        - Level 1: Simple/Consumer (Basic health, general symptoms, common drug side effects).
        - Level 2: Intermediate/Nuanced (Specific clinical scenarios, drug interactions, medical history).
        - Level 3: Advanced/Professional (Highly specialized clinical questions, obscure pathologies, targeted treatment protocols).

        TASK 2: ENCYCLOPEDIA RELEVANCE
        Check if the core subject of the <question> is covered by any of the topics listed in the <encyclopedia_index>.
        - Return 1 if the topic is explicitly listed or heavily implied by a listed category.
        - Return 0 if the topic is completely absent from the index.

        PROCESS:
        Briefly analyze the question inside <thinking> tags. 
        Then, output EXACTLY two tags:
        <difficulty>X</difficulty> (where X is 1, 2, or 3)
        <relevance>Y</relevance> (where Y is 1 or 0)
        </instructions>
        """
    try:
        # Wywołanie lokalnej instancji Ollamy
        response = ollama.chat(
            model='llama3.1:8b', 
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0} # Wymuszamy determinizm
        )
        raw_output = response['message']['content']
        
        # Ekstrakcja danych za pomocą RegEx
        diff_match = re.search(r"<difficulty>([123])</difficulty>", raw_output, re.IGNORECASE)
        rel_match = re.search(r"<relevance>([01])</relevance>", raw_output, re.IGNORECASE)
        
        difficulty = int(diff_match.group(1)) if diff_match else 2
        relevance = int(rel_match.group(1)) if rel_match else 0
        
        return difficulty, relevance
        
    except Exception as e:
        print(f"❌ Błąd lokalnego LLM: {e}")
        return 2, 0

def main():
    print("⚙️ Inicjalizacja skryptu ewaluacyjnego (Wersja Lokalna / Ollama)...")
    
    # 1. Wczytanie listy tematów
    topics_text = load_encyclopedia_topics()
    print(f"📚 Wczytano indeks encyklopedii ({len(topics_text.splitlines())} haseł).")
    
    # 2. Pobranie danych
    df = download_and_sample_dataset()
    question_col = 'Body' if 'Body' in df.columns else df.columns[0]
    df = df.drop_duplicates(subset=[question_col]) # Usuwamy duplikaty, jeśli istnieją
    df = df.dropna(subset=[question_col])
    
    # 3. Przetwarzanie przez LLM
    print("\n🧠 Uruchamianie analizy LLM (Trudność + Trafność)...")
    difficulties = []
    relevances = []
    
    for index, row in df.iterrows():
        q_text = row[question_col]
        print(f"Analiza [{index+1}/{SAMPLE_SIZE}]: {q_text[:60]}...")
        
        diff, rel = assess_question_with_llm(q_text, topics_text)
        difficulties.append(diff)
        relevances.append(rel)
        
    # 4. Zapis do DataFrame i pliku
    df['difficulty_level'] = difficulties
    df['relevance'] = relevances
    
    print("\n📊 STATYSTYKI PRÓBKI:")
    print("--- Rozkład Trudności ---")
    print(df['difficulty_level'].value_counts().sort_index())
    print("\n--- Rozkład Trafności (1=W bazie, 0=Poza bazą) ---")
    print(df['relevance'].value_counts().sort_index())
    
    # Tworzymy folder 'testing', jeśli nie istnieje
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Gotowe! Próbka zapisana do pliku: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()