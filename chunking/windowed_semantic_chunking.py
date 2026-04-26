from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
import re
import torch
import numpy as np
from langchain_community.vectorstores import Chroma

# # 1. Ładowanie modelu raz (Globalnie lub przed pętlą)
# # Sprawdzamy co jest dostępne: MPS (Mac), CUDA (Nvidia) lub CPU
# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

# print(f"Używam urządzenia: {device}")

# embed_model = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={'device': device}
# )

def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def prepare_markdown_for_splitting(text):
    # Czyszczenie
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    headers_to_split_on = [
        ("#", "Medical Topic"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Details"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # If the header becomes a tiny first chunk, consider setting strip_headers to True so the header line doesn’t become a standalone chunk.
    md_header_splits = markdown_splitter.split_text(text)

    return md_header_splits


def advanced_windowed_chunking(text, embed_model, window_size=2, percentile=80):
    # Dzielimy po nowej linii, ignorując puste linie
    units = [line.strip() for line in text.split('\n') if line.strip()]
    
    if len(units) <= window_size:
        return [text]

    # Budujemy okna na podstawie linii
    windows = ["\n".join(units[i : i + window_size]) 
               for i in range(len(units) - window_size + 1)]
    
    # Przekazany model generuje embeddingi
    window_embeddings = embed_model.embed_documents(windows)
    
    distances = []
    for i in range(len(window_embeddings) - 1):
        sim = cosine_similarity([window_embeddings[i]], [window_embeddings[i+1]])[0][0]
        distances.append(1 - sim)

    threshold = np.percentile(distances, percentile)
    
    chunks = []
    start_id = 0
    for i, dist in enumerate(distances):
        if dist > threshold:
            # Łączymy linie w gotowy chunk
            chunks.append("\n".join(units[start_id : i + window_size]))
            start_id = i + window_size
            
    chunks.append("\n".join(units[start_id:]))
    return chunks

def windowed_semantic_chunking(doc, embed_model, window_size=2, percentile=80):
    """
    Przetwarza pojedynczy obiekt Document i zwraca listę mniejszych chunków (jako obiekty Document).
    """
    final_chunks = []
    text_to_split = doc.page_content
    
    if len(text_to_split) > 300: 
        sub_chunk_texts = advanced_windowed_chunking(
            text=text_to_split,
            embed_model=embed_model,
            window_size=window_size,
            percentile=percentile
        )
        
        for text in sub_chunk_texts:
            final_chunks.append(Document(
                page_content=text, 
                metadata=doc.metadata.copy()
            ))
    else:
        final_chunks.append(doc)
        
    return final_chunks

# def save_chunks_to_vector_db(chunks, persist_directory="./medical_db"):
#     """
#     Zapisuje chunki do bazy ChromaDB na dysku.
#     """
#     print(f"Rozpoczynam indeksowanie {len(chunks)} chunków do bazy {persist_directory}...")
    
#     # embed_model to Twój załadowany wcześniej model BGE-M3
#     vector_db = Chroma.from_documents(
#         documents=chunks,
#         embedding=embed_model, # Ten sam model, którego używaliśmy do chunkowania
#         persist_directory=persist_directory
#     )
    
#     print("Baza została zapisana pomyślnie.")
#     return vector_db



# --- URUCHOMIENIE ---

# try:
#     raw_text = load_markdown("encyclopedia/encyclopedia_processed copy.md")
#     print(f"Czy w tekście są nagłówki? {'#' in raw_text}")
#     md_header_splits = prepare_markdown_for_splitting(raw_text)
#     print(f"Liczba sekcji po nagłówkach: {len(md_header_splits)}")

#     chunks = windowed_semantic_chunking(md_header_splits)

#     if chunks:
#         for i in range(0,15):
#             print("\nPrzykładowy chunk:")
#             print(f"Metadane: {chunks[i].metadata}")
#             print(f"Treść: {chunks[i]}...")
#             print(f"\nCałkowita liczba chunków: {len(chunks)}")
#        # vector_db = save_chunks_to_vector_db(chunks)
# except Exception as e:
#     print(f"Wystąpił błąd: {e}")