import spacy
from scispacy.linking import EntityLinker

# 1. Load the SciSpacy base medical model
print("⏳ Loading SciSpacy Medical NLP Model...")
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    print("❌ Error: SciSpacy model 'en_core_sci_sm' not found.")
    print("Please run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz")
    raise

# 2. Add the UMLS Linker to the pipeline
# Note: 'resolve_abbreviations' is crucial for medical texts (e.g., matching "BP" to "Blood Pressure")
print("⏳ Initializing UMLS Entity Linker (This takes a moment)...")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

def extract_umls_cuis(text):
    """
    Analyzes text and returns a set of Concept Unique Identifiers (CUIs).
    Filters out low-confidence matches (below 85%).
    """
    doc = nlp(text)
    cui_set = set()
    
    for entity in doc.ents:
        # Get the top UMLS match for each recognized entity
        if entity._.kb_ents:
            top_cui, match_score = entity._.kb_ents[0]
            # Only accept high-confidence matches (> 85%)
            if match_score > 0.85:
                cui_set.add(top_cui)
                
    return cui_set

# --- Quick Test Execution ---
# This block only runs if you execute this file directly, 
# it will NOT run when imported by retriever.py
if __name__ == "__main__":
    print("\n--- Running Linker Test ---")
    test_text = "The patient presents with high blood pressure and a severe headache."
    cuis = extract_umls_cuis(test_text)
    print(f"Extracted CUIs: {cuis}") 
    # Expected output should contain CUIs for Hypertension and Headache