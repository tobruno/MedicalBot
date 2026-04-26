from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import re
import torch
from langchain_community.vectorstores import Chroma

# 1. Ładowanie modelu raz (Globalnie lub przed pętlą)
# Sprawdzamy co jest dostępne: MPS (Mac), CUDA (Nvidia) lub CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Używam urządzenia: {device}")

embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device}
)

def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def prepare_markdown_for_splitting(text):
    # Czyszczenie
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    headers_to_split_on = [
        ("#", "Notion"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Details"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # If the header becomes a tiny first chunk, consider setting strip_headers to True so the header line doesn’t become a standalone chunk.
    md_header_splits = markdown_splitter.split_text(text)

    return md_header_splits

def markdown_semantic_chunking(md_header_splits):
    # Inicjalizacja chunkera z już załadowanym modelem
    semantic_splitter = SemanticChunker(
        embed_model, 
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=98 
    )

    final_chunks = []
    for doc in md_header_splits:
        # Jeśli sekcja jest krótka (np. sam nagłówek), zostawiamy ją w całości
        if len(doc.page_content) > 300: 
            sub_chunks = semantic_splitter.split_documents([doc])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)
            
    return final_chunks


def save_chunks_to_vector_db(chunks, persist_directory="./medical_db"):
    """
    Zapisuje chunki do bazy ChromaDB na dysku.
    """
    print(f"Rozpoczynam indeksowanie {len(chunks)} chunków do bazy {persist_directory}...")
    
    # embed_model to Twój załadowany wcześniej model BGE-M3
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model, # Ten sam model, którego używaliśmy do chunkowania
        persist_directory=persist_directory
    )
    
    print("Baza została zapisana pomyślnie.")
    return vector_db



# --- URUCHOMIENIE ---
def main():
    try:
        raw_text = load_markdown("encyclopedia/encyclopedia_processed copy.md")
        print(f"Czy w tekście są nagłówki? {'#' in raw_text}")
        md_header_splits = prepare_markdown_for_splitting(raw_text)
        print(f"Liczba sekcji po nagłówkach: {len(md_header_splits)}")

        chunks = markdown_semantic_chunking(md_header_splits)

        if chunks:
            for i in range(0,15):
                print("\nPrzykładowy chunk:")
                print(f"Metadane: {chunks[i].metadata}")
                print(f"Treść: {chunks[i]}...")
                print(f"\nCałkowita liczba chunków: {len(chunks)}")
        # vector_db = save_chunks_to_vector_db(chunks)
    except Exception as e:
        print(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()