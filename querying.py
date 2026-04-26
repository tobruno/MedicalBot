from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import re
import torch
from langchain_chroma.vectorstores import Chroma

# 1. Ładowanie modelu raz (Globalnie lub przed pętlą)
# Sprawdzamy co jest dostępne: MPS (Mac), CUDA (Nvidia) lub CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device}
)


def query_medical_encyclopedia(query, db_directory="./medical_db"):
    # Wczytanie bazy z dysku
    vector_db = Chroma(
        persist_directory=db_directory, 
        embedding_function=embed_model
    )
    
    # Wyszukiwanie 3 najbardziej pasujących fragmentów
    results = vector_db.similarity_search(query, k=1)
    
    return results

# Przykład zapytania:
relevant_docs = query_medical_encyclopedia("How to treat hypertension?")
for doc in relevant_docs:
    print(f"Źródło: {doc.metadata} \nTreść: {doc.page_content[:]}...\n")