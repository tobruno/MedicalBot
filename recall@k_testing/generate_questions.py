import json
import random
import os
import hashlib
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

random.seed(42)

load_dotenv()

groq_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=groq_key)

# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.environ.get("OPENROUTER_API_KEY")
# )

# Konfiguracja
FILES = [
    "chunking/chunks_recursive_raw.json", "chunking/chunks_semantic_raw.json", "chunking/chunks_semantic_raw_v1.json", "chunking/chunks_semantic_raw_v2.json", "chunking/chunks_semantic_raw_70.json", "chunking/chunks_semantic_raw_88.json"
]
SAMPLES_PER_FILE = 100
OUTPUT_DATASET = "recall@k_testing/raw_questions_extended.json"

def get_chunk_id(text):
    """Tworzy unikalne ID dla chunku na podstawie jego tekstu."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def generate_patient_question(chunk_text):
    """Używa LLM do wygenerowania pytania pacjenta na podstawie tekstu."""
    prompt = f"""
    <role>
    You are an average health consumer (a patient or concerned family member). 
    You have no advanced medical training.
    </role>

    <context>
    {chunk_text}
    </context>

    <instructions>
    Based strictly on the <context> provided, formulate ONE moderately complex medical question in English.
    The question should sound like something a real patient would ask a doctor or search for online.
    DO NOT use overly academic jargon unless it's explained in the text.
    The question MUST be fully answerable using ONLY this context.
    
    Output ONLY the question string, nothing else. No quotes, no intro.
    </instructions>
    """
    
    try:
        # response = client.chat.completions.create(
        #     model="meta-llama/llama-3.3-70b-instruct",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.4 # Lekka kreatywność do generowania naturalnych pytań
        # )
        response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4 # Lekka kreatywność do generowania naturalnych pytań
                )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Błąd Groq API: {e}")
        return None

def main():
    dataset = []
    
    for filename in FILES:
        if not os.path.exists(filename):
            print(f"⚠️ Nie znaleziono pliku {filename}. Pomijam.")
            continue
            
        with open(filename, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        print(f"📂 Wczytano {len(chunks)} chunków z {filename}.")
        
        valid_chunks = [c for c in chunks if len(c.get("text", "")) >= 50]
        
        # 2. Losujemy próbkę z przefiltrowanej, "czystej" listy
        sample_size = min(SAMPLES_PER_FILE, len(valid_chunks))
        sampled_chunks = random.sample(valid_chunks, sample_size)
        
        for i, chunk in enumerate(sampled_chunks):
            text = chunk.get("text", "")
            print(f"Generowanie pytania [{i+1}/{sample_size}] dla {filename}...")
            
            question = generate_patient_question(text)
            
            if question:
                dataset.append({
                    "question": question,
                    "expected_chunk_id": get_chunk_id(text),
                    "expected_chunk_text": text, # Zachowujemy tekst do łatwiejszego sprawdzania
                    "source_file": filename,
                    "metadata": chunk.get("metadata", {})
                })
                
    # Zapisujemy nasz "Golden Dataset"
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Gotowe! Zapisano {len(dataset)} pytań testowych do pliku {OUTPUT_DATASET}.")

if __name__ == "__main__":
    main()