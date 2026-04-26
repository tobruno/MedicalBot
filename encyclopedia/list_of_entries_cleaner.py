import re

def clean_entries_list(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    
    for line in lines:
        # 1. Usunięcie białych znaków (spacji, enterów) z początku i końca linii
        text = line.strip()
        
        # Pominięcie pustych linii
        if not text:
            continue
            
        # 2. Usunięcie znaczników typu (jeśli faktycznie są w Twoim pliku tekstownym)
        text = re.sub(r'\\s*', '', text).strip()
        if not text:
            continue

        # 3. WARUNKI FILTROWANIA (Usuwanie śmieci)
        
        # A. Pojedyncze wielkie litery alfabetu (np. "H", "A", "Z")
        if len(text) == 1 and text.isupper():
            continue
            
        # B. Tytuł spisu treści
        if "List of Entries" in text:
            continue
            
        # C. Stopki/nagłówki encyklopedii (z numerami stron z przodu lub z tyłu)
        # Szukamy charakterystycznego, rozstrzelonego ciągu znaków
        if "G A L E E N C Y C L O P E D I A" in text:
            continue
            
        # Jeśli linia przeszła wszystkie filtry, dodajemy ją do czystej listy
        cleaned_lines.append(text)

    # 4. Zapisanie wyczyszczonej listy do nowego pliku
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
        
    print(f"Oczyszczono plik! Zapisano {len(cleaned_lines)} haseł do: {output_file}")

# --- URUCHOMIENIE ---
clean_entries_list("encyclopedia/list_of_entries.txt", "encyclopedia/clean_list_of_entries.txt")