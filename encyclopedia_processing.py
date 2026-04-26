#import pymupdf4llm
import pathlib
#import fitz # PyMuPDF
import re
import glob
import os
import time
import json
from collections import defaultdict


def fix_header_hierarchy(text):
    # 1. Naprawa "płaskiej" struktury: 
    # Szukamy nagłówka ## Cośtam, po którym następuje ## Definition.
    # Używamy lookahead (?=...), aby sprawdzić co jest dalej bez usuwania tego.
    
    # Wzorzec: 
    # \n## + spacja + (tekst nagłówka) + \n
    # (?=\n## Definition) -> za którym bezpośrednio stoi Definicja
    
    pattern = r'\n#+ ([^#\n]+)\n(?=\n#+ Definition)'
    
    # Zamieniamy dowolne ## Hasło na # Hasło, jeśli po nim jest definicja
    fixed_text = re.sub(pattern, r'\n# \1', text)
    
    return fixed_text

def synthesize_missing_header(content):
    """
    Jeśli w treści występuje ## Definition, a brakuje głównego nagłówka #,
    wyciąga podmiot z pierwszego zdania definicji.
    """
    # Lista czasowników definiujących (kotwice)
    defining_verbs = r"\b(\b(is|are|was|were|refers|means|consists|use|uses|occurs|is a|are a)\b"
    
    # 1. Sprawdzamy, czy w tym bloku treści jest ## Definition
    if "## Definition" in content:
        # Szukamy pierwszego zdania po nagłówku Definition
        # pattern: Nagłówek -> puste linie -> [Podmiot] -> [Czasownik]
        match = re.search(rf"## Definition\s*\n+([^.\n]+?)\s+{defining_verbs}", content, re.IGNORECASE)
        
        if match:
            subject = match.group(1).strip()
            # Jeśli podmiot jest sensownej długości (np. do 100 znaków), dodajemy go jako #
            if len(subject) < 100:
                # Wstawiamy nowy nagłówek # na samą górę treści
                content = f"# {subject}\n\n" + content
    return content



import re

import re

def process_medical_rag(full_text):
    # --- KROK 1: WSTĘPNE CZYSZCZENIE I NAPRAWA HIERARCHII ---
    
    # 1.0. Usuwamy przekierowania "see" z poziomu nagłówków ZANIM zaburzą strukturę
    full_text = re.sub(r'(?im)^#+\s+[A-Z][^.!?\n]{1,60}\s+see\s+[A-Z].*?\n', '\n', full_text)
    full_text = re.sub(r'(?im)^#*\s*KEY TERMS\s*$', '\n## KEY TERMS\n', full_text)

    # 1.1. Promujemy ## Tytuł na # Tytuł
    full_text = re.sub(r'\n#+ ([^#\n]+)\n(?=\n#+ Definition)', r'\n# \1', full_text)
    
    # 1.2. Standaryzacja podsekcji do poziomu ##
    standard_subheaders = [
        "Definition", "Description", "Purpose", "Causes and symptoms", 
        "Precautions", "Diagnosis", "Treatment", "Preparation", 
        "Alternative treatment", "Aftercare", "Prognosis", "Risks", 
        "Prevention", "Normal/Abnormal results", "Normal results", 
        "Abnormal results", "Resources", "Key terms", "BOOKS", 
        "PERIODICALS", "ORGANIZATIONS", "Disorders/Conditions", "Tests/Treatments"
    ]
    sub_regex = r"|".join([re.escape(h) for h in standard_subheaders])
    full_text = re.sub(rf'\n#+ ({sub_regex})\s*\n', r'\n## \1\n', full_text, flags=re.IGNORECASE)
    
    # 1.3. Deduplikacja nagłówków głównych
    full_text = re.sub(r'(^|\n)# ([^#\n]+)\n# \2\n', r'\1# \2\n', full_text)

    # --- KROK 2: SYNTEZA BRAKUJĄCYCH NAGŁÓWKÓW ---
    def_split = re.split(r'(\n## Definition)', full_text)
    processed_parts = [def_split[0]]
    defining_verbs = r"\b(is|are|was|were|refers|means|consists|occurs|happens|happen|have|has|is a|are a)\b"
    
    for i in range(1, len(def_split), 2):
        marker, content = def_split[i], def_split[i+1]
        last_sec = processed_parts[-1].strip().split('\n')
        if not any(line.startswith('# ') for line in last_sec[-5:]):
            match = re.search(rf"^\s*([^.\n]+?)\s+{defining_verbs}", content, re.IGNORECASE)
            if match:
                processed_parts[-1] += f"\n# {match.group(1).strip()}\n"
        processed_parts.extend([marker, content])
    
    full_text = "".join(processed_parts)

    # --- KROK 3: PODZIAŁ NA HASŁA I LOKALNE CZYSZCZENIE ---
    sections = re.split(r'(\n# [^#\n]+\n)', full_text)
    final_output = []

    for i in range(1, len(sections), 2):
        header = sections[i]
        content = sections[i+1] if i+1 < len(sections) else ""

        found_terms = []
        
        # A1. Ekstrakcja z dedykowanych bloków "## KEY TERMS" (Bezpieczna)
        kt_matches = re.finditer(r'\n## KEY TERMS\s*\n([\s\S]*?)(?=\n#|\Z)', content, re.IGNORECASE)
        for kt_match in kt_matches:
            kt_block = kt_match.group(1)
            paragraphs = re.split(r'\n\s*\n', kt_block)
            for p in paragraphs:
                p = p.strip()
                if re.match(r'^[A-Z][^\n]{1,80}?[—–\-]\s*[A-Z]', p):
                    found_terms.append(p)
            content = content.replace(kt_match.group(0), "\n[MARKER]\n")

        # A2. Ekstrakcja "osieroconych" pojęć (Bezpieczna)
        dash_chars = r'—–\-'
        # Wzorzec usztywniony - pobiera max 8 linii tekstu po myślniku, aby zapobiec pętlom
        orphaned_pattern = rf'(?m)^([A-Z][^{dash_chars}\n.!?]{{1,80}}[{dash_chars}]\s*[A-Z][^\n]*(?:\n(?!\s*\n|#)[^\n]*){{0,8}})'
        
        orphaned_matches = list(re.finditer(orphaned_pattern, content))
        for match in orphaned_matches:
            term_text = match.group(1)
            found_terms.append(term_text)
            content = content.replace(term_text, "[MARKER]\n")

        # B. NAPRAWA LIST I ARTEFAKTÓW
        content = re.sub(r'(?m)^.*?\((?:Illustration|Photograph|Photo|Figure) by [^)]+\.?\)\s*$', '[MARKER]', content, flags=re.IGNORECASE)
        content = re.sub(r'\!\[.*?\](?:\(.*?\))?', '[MARKER]', content)
        content = re.sub(r'^[ \t]*[-*]?\s*!\s*', r'- ', content, flags=re.MULTILINE)

        # C. PRECYZYJNE USUWANIE PRZEKIEROWAŃ "see" w treści
        content = re.sub(r'^[#\s]*[A-Z][^.!?]{1,60}\s+see\s+[A-Z].*$', '[MARKER]', content, flags=re.MULTILINE | re.IGNORECASE)

        # D. POZOSTAŁY SZUM (Tabele, Bibliografia)
        content = re.sub(r'^.*\|.*$', '[MARKER]', content, flags=re.MULTILINE)
        content = re.sub(r'\n[^#\n]+?\(.*?Reproduced by permission\.\)', "\n[MARKER]\n", content)
        
        noise_headers = ["Resources", "PERIODICALS", "BOOKS", "ORGANIZATIONS", "OTHER"]
        noise_regex = r"|".join(noise_headers)
        content = re.sub(rf'(?im)\n##\s+(?:{noise_regex})\b.*?(?=\n## |\Z)', '\n[MARKER]\n', content, flags=re.DOTALL)

        # E. SCALANIE I LECZENIE (HEALING) - CAŁKOWICIE NOWA, BEZPIECZNA METODA
        # 1. Kompresujemy wszystkie zgrupowane markery i puste znaki wokół nich w jeden, czysty znacznik:
        content = re.sub(r'\s*(?:\[MARKER\]\s*)+', ' [MARKER] ', content)
        
        # 2. Bezpiecznie scalane małe litery rozdzielone zredukowanym znacznikiem:
        content = re.sub(r'([a-z,;])\s*\[MARKER\]\s*([a-z])', r'\1 \2', content)
        
        # 3. Standardowe usuwanie sierot po nowych liniach:
        content = re.sub(r'([a-z,;])\s*\n+\s*([a-z])', r'\1 \2', content)
        
        # 4. Całkowite pozbycie się resztek:
        clean_content = content.replace("[MARKER]", "").strip()
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
        clean_content = re.sub(r'-\s?-\s?', '- ', clean_content)

        # F. DOKLEJENIE KEY TERMS
        if found_terms:
            cleaned_terms = []
            for t in found_terms:
                t = t.strip()
                t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
                cleaned_terms.append(t)
                
            unique_terms = sorted(list(set(cleaned_terms)))
            clean_content += "\n\n## KEY TERMS\n\n" + "\n\n".join(unique_terms)

        final_output.append(header + clean_content + "\n\n")

    return re.sub(r'(^|\n)# ([^#\n]+)\n# \2\n', r'\1# \2\n', "".join(final_output))


def process_directory_step_by_step(input_pattern, output_file):
    # 1. Pobieranie i sortowanie plików
    file_list = glob.glob(input_pattern)
    file_list.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)])
    
    total_files = len(file_list)
    print(f"Znaleziono {total_files} plików do przetworzenia.\n" + "-"*40)

    all_processed_chunks = []
    
    # 2. Pętla przez każdy plik z osobna
    for idx, file_path in enumerate(file_list, 1):
        print(f"[{idx}/{total_files}] Przetwarzanie: {file_path} ... ", end="", flush=True)
        start_time = time.time()
        
        # Wczytanie
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        # Przetworzenie TYLKO TEGO JEDNEGO PLIKU
        processed_text = process_medical_rag(raw_text)
        all_processed_chunks.append(processed_text)
        
        # Obliczenie czasu
        elapsed = time.time() - start_time
        print(f"Gotowe w {elapsed:.2f} s")

    # 3. Łączenie wszystkich przetworzonych plików
    print("-" * 40)
    print("Łączenie plików w jedną spójną bazę...")
    merged_text = "\n\n".join(all_processed_chunks)

    # 4. Finalna, lekka deduplikacja na stykach plików
    print("Ostateczne polerowanie styków dokumentu...")
    merged_text = re.sub(r'(^|\n)# ([^#\n]+)\n# \2\n', r'\1# \2\n', merged_text)
    merged_text = re.sub(r'\n{3,}', '\n\n', merged_text)

    # 5. Zapis
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_text)
    
    print(f"\nSukces! Finalny plik RAG zapisano jako: {output_file}")



def merge_and_transform(input_pattern):
    # 1. Znajdź wszystkie pliki pasujące do wzorca (np. data/part_*/part_*.md)
    file_list = glob.glob(input_pattern)
    
    # 2. Sortowanie plików (ważne, aby part_1 był przed part_10)
    # Sortujemy naturalnie po numerze w ścieżce
    file_list.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)])
    
    print(f"Znaleziono {len(file_list)} plików do połączenia...")

    all_raw_text = ""
    
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Dodajemy nową linię przed każdym plikiem, aby nagłówki się nie skleiły
            all_raw_text += "\n" + f.read()
            print(f"Wczytano: {file_path}")

    return all_raw_text


# with open("data/part_1/part_1.md", 'r', encoding='utf-8') as f:
#     text = f.read()
# text = process_medical_rag(text)
# output_file = "data/part_1/part_1_fixed.md"
# with open(output_file, "w", encoding='utf-8') as f:
#     f.write(text)
# print(f"Sukces! Finalny plik zapisano jako: {output_file}")


# text = merge_and_transform("data/part_*/part_*.md")
# fixed_text = process_medical_rag(text)

# output_file = "encyclopedia/encyclopedia_processed_v1.md"

# with open(output_file, "w", encoding='utf-8') as f:
#     f.write(fixed_text)
# print(f"Sukces! Finalny plik zapisano jako: {output_file}")



# --- URUCHOMIENIE ---
process_directory_step_by_step("data/part_*/part_*.md", "encyclopedia/encyclopedia_processed_v1.md")