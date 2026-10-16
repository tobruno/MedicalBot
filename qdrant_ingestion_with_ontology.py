import json
import hashlib
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# WYRZUCAMY fastembed I UŻYWAMY OFICJALNEJ BIBLIOTEKI BAAI
from FlagEmbedding import BGEM3FlagModel

# --- 1. KONFIGURACJA ETAPU 2 ---
INPUT_JSON = "chunking/chunks_semantic_raw_88.json" #"chunking/chunks_recursive_contextualised_with_ontology.json" 
QDRANT_PATH = "./medical_qdrant_db"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "semantic_raw_88" #if "with_ontology" in INPUT_JSON else "semantic_raw"

print(f"🚀 Inicjalizacja: Baza={QDRANT_PATH}")
print(f"📂 Cel: Kolekcja '{COLLECTION_NAME}' (Źródło: {INPUT_JSON})")

# Ładowanie oficjalnego modelu BGE-M3 (obsługuje 8192 tokeny)
print("⏳ Ładowanie oficjalnego modelu BGE-M3 (Dense + Sparse w jednym)...")
# use_fp16=False jest bezpieczniejsze dla CPU/Mac
bge_m3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, num_workers=os.cpu_count())

# --- 2. ZASILANIE QDRANT ---
def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )}
    )
    
    indexed_fields = ["MedicalTopic", "Section", "Subsection", "Details", "umls_cuis"]
    for field in indexed_fields:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=f"metadata.{field}",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        
    return client


def ingest_to_qdrant(client, data_items, batch_size=64): # Zwiększono batch_size
    print(f"📤 Indeksowanie hybrydowe {len(data_items)} punktów...")
    
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i : i + batch_size]
        texts = [item["text"] for item in batch]
        metadatas = [item["metadata"] for item in batch]
        
        # Generowanie Wektorów
        embeddings = bge_m3_model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense_vecs = embeddings['dense_vecs']
        sparse_vecs_list = embeddings['lexical_weights']
        
        points = []
        for j in range(len(batch)):
            chunk_hash = hashlib.md5(texts[j].encode('utf-8')).hexdigest()
            point_id = str(uuid.UUID(chunk_hash))
            
            sparse_dict = sparse_vecs_list[j]
            # BGE-M3 zwraca klucze jako stringi, Qdrant wymaga int
            sparse_indices = [int(k) for k in sparse_dict.keys()]
            sparse_values = [float(v) for v in sparse_dict.values()]
            
            # Tworzenie obiektu PointStruct - to jest kluczowe
            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vecs[j].tolist(),
                    "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload={
                    "text": texts[j], 
                    "metadata": metadatas[j]
                }
            ))
        
        # Używamy upsert, ale w większych paczkach
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        print(f"   Zasilono: {i + len(batch)}/{len(data_items)}")

# --- URUCHOMIENIE ETAPU 2 ---
def main():
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            data_items = json.load(f)
            
        q_client = init_qdrant()
        ingest_to_qdrant(q_client, data_items)
        
        print(f"\n✨ Sukces! Baza Qdrant w pełni gotowa do zaawansowanego hybrydowego RAG.")
        
    except FileNotFoundError:
        print(f"❌ Błąd: Nie znaleziono pliku '{INPUT_JSON}'. Uruchom najpierw Etap 1.")
    except Exception as e:
        print(f"❌ Błąd krytyczny Etapu 2: {e}")

if __name__ == "__main__":
    main()