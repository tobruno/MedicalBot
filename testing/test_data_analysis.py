import pandas as pd
from datasets import load_dataset

# --- KONFIGURACJA ---
DATASET_NAME = "truehealth/liveqa"
INPUT_FILE = "medredqa/medredqa_train.csv"

def analyze_medredqa_sample():
        
    print(f"📥 Pobieranie datasetu '{INPUT_FILE}'...")
    dataset = pd.read_csv(INPUT_FILE)
    dataset = dataset[dataset['Occupation'] != 'Medical Student']
    
    # Usuwanie wierszy, gdzie 'message' jest puste, None lub składa się tylko ze spacji
    print("🧹 Usuwanie pustych wierszy w kolumnie 'Body'...")
    dataset = dataset.dropna(subset=['Body'])
    dataset = dataset.dropna(subset=['Response'])

    
    dataset = dataset[dataset['Response Score']>20]
    dataset = dataset.loc[dataset['Body'].str.contains('EDIT|UPDATE|CORRECTION|Sex:|Sex M|Sex F|photo|image', case=False, na=False) == False]
    

def analyze_liveqa_sample():
    print(f"📥 Pobieranie datasetu '{DATASET_NAME}' z Hugging Face...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    # Usuwanie wierszy, gdzie 'message' jest puste, None lub składa się tylko ze spacji
    print("🧹 Usuwanie pustych wierszy w kolumnie 'message'...")
    dataset = dataset.filter(
        lambda example: example['message'] is not None and str(example['message']).strip() != ""
    )
    #dataset = pd.read_csv("testing/liveqa_sampled_evaluated.csv")