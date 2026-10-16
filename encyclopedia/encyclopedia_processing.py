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
    
    pattern = r'\n#+ ([^#\n]+)\n(?=\n#+ Definition)'
    
    # Zamieniamy dowolne ## Hasło na # Hasło, jeśli po nim jest definicja
    fixed_text = re.sub(pattern, r'\n# \1', text)
    
    return fixed_text

def synthesize_missing_header(content):
    """
    Jeśli w treści występuje ## Definition, a brakuje głównego nagłówka #,
    wyciąga podmiot z pierwszego zdania definicji.
    """
    defining_verbs = r"\b(\b(is|are|was|were|refers|means|consists|use|uses|occurs|is a|are a)\b"
    
    if "## Definition" in content:
        match = re.search(rf"## Definition\s*\n+([^.\n]+?)\s+{defining_verbs}", content, re.IGNORECASE)
        
        if match:
            subject = match.group(1).strip()
            if len(subject) < 100:
                content = f"# {subject}\n\n" + content
    return content


def process_medical_rag(full_text):
    # --- KROK 1: WSTĘPNE CZYSZCZENIE I NAPRAWA HIERARCHII ---
    
    full_text = re.sub(r'(?im)^#+\s+[A-Z][^.!?\n]{1,60}\s+see\s+[A-Z].*?\n', '\n', full_text)
    full_text = re.sub(r'(?im)^#*\s*KEY TERMS\s*$', '\n## KEY TERMS\n', full_text)

    full_text = re.sub(r'\n#+ ([^#\n]+)\n(?=\n#+ Definition)', r'\n# \1', full_text)
    
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

        # E. SCALANIE I LECZENIE (HEALING)
        content = re.sub(r'\s*(?:\[MARKER\]\s*)+', ' [MARKER] ', content)
        content = re.sub(r'([a-z,;])\s*\[MARKER\]\s*([a-z])', r'\1 \2', content)
        content = re.sub(r'([a-z,;])\s*\n+\s*([a-z])', r'\1 \2', content)
        
        # 4. Całkowite pozbycie się resztek i modyfikacja struktur:
        clean_content = content.replace("[MARKER]", "").strip()
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
        
        # 5. Ściskanie podwójnych linii PRZED listą oraz MIĘDZY jej punktami
        clean_content = re.sub(r'([^\n])\n{2,}(?=[ \t]*-[ \t])', r'\1\n', clean_content)
        
        clean_content = re.sub(r'-\s?-\s?', '- ', clean_content)

        # F. DOKLEJENIE KEY TERMS (Z pojedynczą nową linią)
        if found_terms:
            cleaned_terms = []
            for t in found_terms:
                t = t.strip()
                t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
                cleaned_terms.append(t)
                
            unique_terms = sorted(list(set(cleaned_terms)))
            # NOWOŚĆ: używamy "\n".join() zamiast "\n\n".join() dla zwartej struktury słowniczka
            clean_content += "\n\n## KEY TERMS\n\n" + "\n".join(unique_terms)

        final_output.append(header + clean_content + "\n\n")

    return re.sub(r'(^|\n)# ([^#\n]+)\n# \2\n', r'\1# \2\n', "".join(final_output))


def process_directory_step_by_step(input_pattern, output_file):
    file_list = glob.glob(input_pattern)
    file_list.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)])
    
    total_files = len(file_list)
    print(f"Znaleziono {total_files} plików do przetworzenia.\n" + "-"*40)

    all_processed_chunks = []
    
    for idx, file_path in enumerate(file_list, 1):
        print(f"[{idx}/{total_files}] Przetwarzanie: {file_path} ... ", end="", flush=True)
        start_time = time.time()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        processed_text = process_medical_rag(raw_text)
        all_processed_chunks.append(processed_text)
        
        elapsed = time.time() - start_time
        print(f"Gotowe w {elapsed:.2f} s")

    print("-" * 40)
    print("Łączenie plików w jedną spójną bazę...")
    merged_text = "\n\n".join(all_processed_chunks)

    print("Ostateczne polerowanie styków dokumentu...")
    merged_text = re.sub(r'(^|\n)# ([^#\n]+)\n# \2\n', r'\1# \2\n', merged_text)
    merged_text = re.sub(r'\n{3,}', '\n\n', merged_text)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_text)
    
    print(f"\nSukces! Finalny plik RAG zapisano jako: {output_file}")


def merge_and_transform(input_pattern):
    file_list = glob.glob(input_pattern)
    file_list.sort(key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)])
    
    print(f"Znaleziono {len(file_list)} plików do połączenia...")

    all_raw_text = ""
    
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_raw_text += "\n" + f.read()
            print(f"Wczytano: {file_path}")

    return all_raw_text

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    process_directory_step_by_step("encyclopedia/data/part_*/part_*.md", "encyclopedia/encyclopedia_processed.md")