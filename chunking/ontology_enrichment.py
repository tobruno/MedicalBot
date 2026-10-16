import json
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# WE IMPORT THE CORE LOGIC FROM YOUR EXISTING FILE
from retriever.ontology_linker import extract_umls_cuis

# --- CONFIGURATION ---
INPUT_JSON = "chunking/chunks_semantic_with_context_v1.json"
OUTPUT_JSON = "chunking/chunks_semantic_contextualised_with_ontology_v1.json"

def enrich_chunks_with_cuis():
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find {INPUT_JSON}.")
        return

    print(f"🚀 Starting Ontology Enrichment for {len(chunks)} chunks...")
    start_time = time.time()
    
    # Iterate through every existing chunk
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        
        # WE USE THE IMPORTED FUNCTION HERE!
        # No need to duplicate the SciSpacy logic.
        extracted_cuis = extract_umls_cuis(text)
        
        # Add the extracted CUIs into the chunk's metadata
        chunk["metadata"]["umls_cuis"] = list(extracted_cuis)
        
        # Progress tracker
        if (i + 1) % 500 == 0:
            print(f"   [Enrichment] Processed {i + 1}/{len(chunks)} chunks...")
            
    print("💾 Saving enriched dataset...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
        
    elapsed_time = round(time.time() - start_time, 2)
    print(f"✨ Success! Saved enriched chunks to {OUTPUT_JSON}.")
    print(f"⏱️ Total time taken: {elapsed_time} seconds.")

if __name__ == "__main__":
    enrich_chunks_with_cuis()