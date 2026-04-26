import json
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings # Darmowe wektory lokalnie

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevance,
)

# Load API keys
load_dotenv()

RESULTS_FILE = "results_with_umls.json"

def main():
    print(f"📊 Loading experiment data from {RESULTS_FILE}...")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    # 1. Filter out Safe Abstentions (Crucial for Medical RAG)
    # We don't evaluate answer_relevance or faithfulness if the bot correctly refused to answer!
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    safe_abstention_count = 0
    system_error_count = 0
    
    for item in raw_data:
        ans = item["answer"]
        if "insufficient clinical data" in ans.lower() or "does not contain information" in ans.lower():
            safe_abstention_count += 1
            continue
        if "⚠️" in ans:
            system_error_count += 1
            continue
            
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(item["answer"])
        eval_data["contexts"].append(item["contexts"])

    print(f"\n--- Pre-Evaluation Stats ---")
    print(f"Total Questions: {len(raw_data)}")
    print(f"Safe Abstentions (True Negatives): {safe_abstention_count}")
    print(f"System Errors (Format failures): {system_error_count}")
    print(f"Valid Answers to Evaluate: {len(eval_data['question'])}")
    print("----------------------------\n")

    if len(eval_data['question']) == 0:
        print("No valid answers to evaluate. Exiting.")
        return

    # 2. Convert to HuggingFace Dataset format required by RAGAS
    dataset = Dataset.from_dict(eval_data)

    print("⚖️ Initializing OpenRouter LLM as the Judge...")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    # 1. Konfiguracja LLM przez OpenRouter
    judge_llm = ChatOpenAI(
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct:free", # Zmień na dowolny model z OpenRouter
        temperature=0.0
        # OpenRouter zaleca dodanie tych nagłówków (opcjonalne, ale dobra praktyka)
        # default_headers={
        #     "HTTP-Referer": "https://twojastrona.pl", # Możesz wpisać cokolwiek
        #     "X-Title": "MedicalBot Evaluation",
        # }
    )

    # 2. Konfiguracja modelu wektorowego (Embeddings)
    # UWAGA: RAGAS potrzebuje też modelu do wektorów. OpenRouter służy głównie do generowania tekstu.
    # Najlepiej i najtaniej jest użyć Twojej lokalnej Ollamy do wektorów (to nic nie kosztuje!)
    judge_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Run the Evaluation
    print("🚀 Starting RAGAS Evaluation... This may take a few minutes.")
    metrics = [
        faithfulness,      # Checks if the generated answer hallucinates outside contexts
        answer_relevance,  # Checks if the answer actually addresses the question
        context_precision  # Evaluates the Retriever's sorting quality
    ]
    # Note: context_recall is omitted here because it technically requires a "ground_truth" column.
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    # 5. Display and Save Results
    df = result.to_pandas()
    print("\n✅ Evaluation Complete!")
    print("\n--- AGGREGATE SCORES ---")
    print(result)
    
    csv_name = RESULTS_FILE.replace(".json", "_evaluation.csv")
    df.to_csv(csv_name, index=False)
    print(f"\n💾 Detailed question-by-question report saved to {csv_name}")

if __name__ == "__main__":
    main()