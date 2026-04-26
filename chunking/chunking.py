import os
import json
import re
import ollama
import hashlib
import torch
import time
from groq import Groq
from collections import defaultdict
import concurrent.futures
import threading

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Importujemy naszą poprawioną funkcję z modułu obok
from chunking.windowed_semantic_chunking import windowed_semantic_chunking

#groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- 1. KONFIGURACJA ETAPU 1 ---
INPUT_FILE = "encyclopedia/encyclopedia_processed.md"

# 🎛️ PRZEŁĄCZNIKI EKSPERYMENTÓW
USE_CONTEXTUALIZATION = True 
CHUNKING_METHOD = "semantic"  # Wybierz: "recursive" lub "semantic"

# Konfiguracja dla metody recursive
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# Konfiguracja dla metody semantic
WINDOW_SIZE = 3
PERCENTILE = 92

# Dynamiczne nazwy plików wyjściowych
OUTPUT_JSON = f"chunking/chunks_{CHUNKING_METHOD}_{'with_context' if USE_CONTEXTUALIZATION else 'raw'}.json"
CACHE_PATH = f"chunking/context_cache_{CHUNKING_METHOD}.json"

# Inicjalizacja modelu TYLKO jeśli jest wymagany do cięcia semantycznego
embed_model = None
if CHUNKING_METHOD == "semantic":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🧠 Ładowanie modelu BGE-M3 (Urządzenie: {device})...")
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})

# --- 2. FUNKCJE PRZETWARZANIA TEKSTU ---
def load_and_split_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    headers_to_split_on = [
        ("#", "Medical Topic"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Details"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    return markdown_splitter.split_text(text)

def create_knowledge_map(md_header_splits):
    k_map = defaultdict(str)
    for doc in md_header_splits:
        notion = doc.metadata.get('Medical Topic', 'Unknown')
        k_map[notion] += f"\n\n{doc.page_content}"
    return k_map

# # --- 3. MECHANIZM KONTEKSTUALIZACJI GROQ---
# def generate_context_with_groq(notion_text, chunk_text, metadata):
#     notion = metadata.get('Medical Topic', 'Medical Topic')
#     section = metadata.get('Section', 'General')
    
#     prompt = f"""
#     The following medical document is about: {notion}.
#     <context>
#     {notion_text[:5000]} 
#     </context>
#     Here is the chunk we want to situate within the whole document:
#     <chunk_to_contextualize>
#     {chunk_text}
#     </chunk_to_contextualize>

#     Instruction: Give a short (50-70 words) succinct context to situate this chunk within the overall topic of "{notion}" for the purposes of improving search retrieval of the chunk.
#     Answer ONLY with the succinct context and nothing else.
#     """
#     try:
#         # Wywołanie API Groq (Używamy ultraszybkiego modelu Llama 3.1 8B)
#         chat_completion = groq_client.chat.completions.create(
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             model="llama-3.1-8b-instant", # Możesz też użyć "llama3-70b-8192"
#             temperature=0.1, # Niska temperatura = bardziej przewidywalny kontekst
#         )
#         return chat_completion.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"⚠️ Błąd Groq API: {e}")
#         return f"This section discusses {notion} in the context of {section}."

# --- 3. MECHANIZM KONTEKSTUALIZACJI ---
def generate_context_with_ollama(notion_text, chunk_text, metadata):
    notion = metadata.get('Medical Topic', 'Medical Topic')
    section = metadata.get('Section', 'General')
    
    prompt = f"""
    The following medical document is about: {notion}.
    <context>
    {notion_text[:5000]} 
    </context>
    Here is the chunk we want to situate within the whole document:
    <chunk_to_contextualize>
    {chunk_text}
    </chunk_to_contextualize>

    Instruction: Give a short (50-70 words) succinct context to situate this chunk within the overall topic of "{notion}" for the purposes of improving search retrieval of the chunk.
    Answer ONLY with the succinct context and nothing else.
    """
    try:
        #response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except Exception:
        return f"This section discusses {notion} in the context of {section}."

# Globalna kłódka dla bezpieczeństwa pliku cache
cache_lock = threading.Lock()

def process_single_chunk_task(sub_chunk, notion, notion_full_text, use_context, cache):
    """Pomocnicza funkcja dla pojedynczego wątku."""
    final_text = sub_chunk.page_content
    
    if use_context:
        unique_string = (sub_chunk.page_content + notion).encode('utf-8')
        chunk_key = hashlib.md5(unique_string).hexdigest()
        
        # Odczyt ze słownika (jest bezpieczny w Pythonie)
        if chunk_key in cache:
            final_text = cache[chunk_key]
        else:
            # 1. Pytamy Ollamę (tu wątek pracuje)
            context = generate_context_with_ollama(notion_full_text, sub_chunk.page_content, sub_chunk.metadata)
            final_text = f"Context: {context}\n\nContent: {sub_chunk.page_content}"
            
            # 2. Blokujemy dostęp na ułamek sekundy, by zapisać cache
            with cache_lock:
                cache[chunk_key] = final_text
                # Zapisujemy rzadziej (np. co 50), by nie katować dysku przy 4 wątkach
                if len(cache) % 50 == 0:
                    with open(CACHE_PATH, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False)
                        
    return {
        "text": final_text,
        "metadata": sub_chunk.metadata
    }


def process_chunks(md_header_splits, knowledge_map, use_context, method=CHUNKING_METHOD):
    print(f"🧩 Metoda: [{method.upper()}]. Kontekstualizacja LLM: [{'WŁĄCZONA' if use_context else 'WYŁĄCZONA'}]")
    
    if use_context and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    
    # 1. Najpierw generujemy i zbieramy wszystkie chunki "na sucho"
    print("⏳ Tnę tekst (przygotowuję zadania dla LLM)...")
    tasks = []
    
    for doc in md_header_splits:
        notion = doc.metadata.get('Medical Topic', 'Unknown')
        notion_full_text = knowledge_map.get(notion, "")
        
        if method == "semantic":
            sub_chunks = windowed_semantic_chunking(doc, embed_model, window_size=WINDOW_SIZE, percentile=PERCENTILE)
        else:
            sub_chunks = recursive_splitter.split_documents([doc])
            
        for sub_chunk in sub_chunks:
            # Tworzymy pakiet danych do wysłania w kosmos (do wątku)
            tasks.append((sub_chunk, notion, notion_full_text, use_context, cache))
            
    print(f"🚀 Pocięto na {len(tasks)} chunków. Startuję 4 równoległe wątki Ollamy!")

    processed_data = []
    
    # 2. Wysyłamy zadania do 4 "robotników" naraz
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Odpalamy wszystkie zadania
        futures = [executor.submit(process_single_chunk_task, *task) for task in tasks]
        
        # Pasek postępu - odbieramy gotowe kawałki
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            processed_data.append(future.result())
            
            # Wypisuj postęp co 100 ukończonych chunków
            if (i + 1) % 100 == 0:
                print(f"   [Wielowątkowość] Przetworzono {i + 1}/{len(tasks)}...")

    # Finalny zapis resztek cache
    if use_context:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
            
    return processed_data

# WARIANT PRZETWARZANIA W POJEDYNCZYM WĄTKU (BEZ KONTEKSTUALIZACJI WIELOWĄTKOWEJ)
# def process_chunks(md_header_splits, knowledge_map, use_context, method=CHUNKING_METHOD):
#     print(f"🧩 Metoda cięcia: [{method.upper()}]. Kontekstualizacja LLM: [{'WŁĄCZONA' if use_context else 'WYŁĄCZONA'}]")
    
#     if use_context and os.path.exists(CACHE_PATH):
#         with open(CACHE_PATH, "r", encoding="utf-8") as f:
#             cache = json.load(f)
#     else:
#         cache = {}

#     recursive_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         add_start_index=True,
#     )
    
#     processed_data = []
    
#     for i, doc in enumerate(md_header_splits):
#         notion = doc.metadata.get('Medical Topic', 'Unknown')
#         notion_full_text = knowledge_map.get(notion, "")
        
#         # 🔀 WYBÓR METODY CHUNKOWANIA
#         if method == "semantic":
#             sub_chunks = windowed_semantic_chunking(doc, embed_model, window_size=WINDOW_SIZE, percentile=PERCENTILE)
#         elif method == "recursive":
#             sub_chunks = recursive_splitter.split_documents([doc])
#         else: 
#             sub_chunks = recursive_splitter.split_documents([doc])
            
#         print(f"[{i+1}/{len(md_header_splits)}] Przetwarzanie '{notion}' ({len(sub_chunks)} chunki)...")
        
#         for sub_chunk in sub_chunks:
#             final_text = sub_chunk.page_content
            
#             if use_context:
#                 unique_string = (sub_chunk.page_content + notion).encode('utf-8')
#                 chunk_key = hashlib.md5(unique_string).hexdigest()
                
#                 if chunk_key in cache:
#                     final_text = cache[chunk_key]
#                 else:
#                     #context = generate_context_with_ollama(notion_full_text, sub_chunk.page_content, sub_chunk.metadata)
#                     # Zamiast: context = generate_context_with_ollama(...)
#                     context = generate_context_with_groq(notion_full_text, sub_chunk.page_content, sub_chunk.metadata)
#                     #time.sleep(1) # Czeka 2.1 sekundy, co daje max ~28 zapytań na minutę (Groq ma limit 30 zapytań/minutę, ale zostawiamy trochę buforu)
#                     final_text = f"Context: {context}\n\nContent: {sub_chunk.page_content}"
#                     cache[chunk_key] = final_text
                    
#                     if len(cache) % 20 == 0:
#                         with open(CACHE_PATH, "w", encoding="utf-8") as f:
#                             json.dump(cache, f, ensure_ascii=False)

#             processed_data.append({
#                 "text": final_text,
#                 "metadata": sub_chunk.metadata
#             })
            
#     # Finalny zapis resztek cache
#     if use_context:
#         with open(CACHE_PATH, "w", encoding="utf-8") as f:
#             json.dump(cache, f, ensure_ascii=False)
            
#     return processed_data

# --- URUCHOMIENIE ETAPU 1 ---
def main():
    try:
        md_header_splits = load_and_split_markdown(INPUT_FILE)
        knowledge_map = create_knowledge_map(md_header_splits)
        
        final_data = process_chunks(md_header_splits, knowledge_map, USE_CONTEXTUALIZATION)
        
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n✨ Etap 1 zakończony! Zapisano {len(final_data)} chunków do pliku: {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"❌ Błąd krytyczny Etapu 1: {e}")

if __name__ == "__main__":
    main()