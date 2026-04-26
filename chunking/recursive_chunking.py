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
        ("#", "Medical Topic"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Details"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # If the header becomes a tiny first chunk, consider setting strip_headers to True so the header line doesn’t become a standalone chunk.
    md_header_splits = markdown_splitter.split_text(text)

    return md_header_splits

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ollama
import time
from langchain_core.documents import Document

# --- NOWA FUNKCJA: KONTEKSTUALIZACJA ---

def generate_context_with_ollama(notion_text, chunk_text, metadata):
    notion = metadata.get('Medical Topic', 'Medical Topic')
    section = metadata.get('Section', 'General Info')

    prompt = f"""
    <whole_article_context>
    {notion_text[:8000]} 
    </document_context>
    
    <specific_chunk>
    {chunk_text}
    </specific_chunk>

    Instruction: The chunk above comes from the section '{section}' within the article '{notion}'.
    Provide a concise (max 50 words) context that connects this chunk to the main topic of {notion}.
    Explain what this specific part refers to, so it's clear even if read in isolation.
    Respond ONLY with the context.
    """
    
    response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()
# --- ZMODYFIKOWANE CHUNKOWANIE ---

def recursive_markdown_chunking(md_header_splits, chunk_size=1000, chunk_overlap=300, use_contextualization=False):
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    final_chunks = []
    
    for i, doc in enumerate(md_header_splits):
        # Zachowujemy pełny tekst sekcji (rodzica) jako kontekst
        parent_text = doc.page_content
        
        # Tworzymy mniejsze chunki
        sub_chunks = recursive_splitter.split_documents([doc])
        
        if use_contextualization:
            print(f"Trwa kontekstualizacja sekcji {i+1}/{len(md_header_splits)}: {doc.metadata.get('Section', 'N/A')}")
            for sub_chunk in sub_chunks:
                # Generujemy kontekst
                context_summary = generate_context_with_ollama(parent_text, sub_chunk.page_content, sub_chunk.metadata)
                
                # Doklejamy kontekst (Contextual Prepending)
                sub_chunk.page_content = f"Context: {context_summary}\n\nContent: {sub_chunk.page_content}"
                final_chunks.append(sub_chunk)
        else:
            final_chunks.extend(sub_chunks)
            
    return final_chunks

# Wywołanie:
# chunks = recursive_markdown_chunking(md_header_splits, chunk_size=1200, chunk_overlap=200)


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

def main():
    try:
        raw_text = load_markdown("encyclopedia/encyclopedia_processed copy.md")
        md_header_splits = prepare_markdown_for_splitting(raw_text)

        # Przekazujemy flagę do funkcji
        chunks = recursive_markdown_chunking(
            md_header_splits, 
            use_contextualization=USE_LLM_CONTEXT
        )

        if chunks:
            # Wyświetl pierwszy wzbogacony chunk
            print("\nPrzykładowy wzbogacony chunk:")
            print(f"Treść: {chunks[15].page_content[:500]}...")
            
            # save_chunks_to_vector_db(chunks)

    except Exception as e:
        print(f"Wystąpił błąd: {e}")


# --- URUCHOMIENIE ---
if __name__ == "__main__":
    USE_LLM_CONTEXT = True
    main()
