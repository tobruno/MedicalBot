import ollama
import re
import os
from groq import Groq 
from openai import OpenAI
from dotenv import load_dotenv

# Inicjalizacja klienta wskazująca na OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)


load_dotenv()
groq_key = os.environ.get("GROQ_API_KEY")
#client = Groq(api_key=groq_key)

def generate_grounded_response(user_query, retrieved_chunks):
    """
    Generates a safe response and returns data structured for RAGAS evaluation.
    """
    context_body = "\n\n".join([f"<chunk id=\"{i+1}\">\n{chunk['text']}\n</chunk>" for i, chunk in enumerate(retrieved_chunks)])
    
    prompt = f"""
        <role>
        You are a Strict Clinical AI Assistant operating under a Pessimistic Grounding Architecture. 
        Your highest priority is patient safety and absolute adherence to the provided documents.
        You are NOT a doctor. You cannot diagnose, prescribe treatment, or give personalized medical advice.
        </role>

        <context>
        {context_body}
        </context>

        <patient_query>
        {user_query}
        </patient_query>

        <instructions>
        1. STRICT GROUNDING: Answer the patient_query using ONLY facts explicitly stated in the <context> tags. You must completely suppress your internal parametric knowledge.
        2. ABSTENTION POLICY: If the <context> is irrelevant, or if it lacks sufficient clinical depth to answer the specific question safely, you MUST abort the generation.
        3. CITATION RULE: Every single sentence in your final answer that contains a medical claim MUST end with a citation to the specific chunk (e.g., "Aspirin can cause gastric irritation [Chunk 2].").
        4. DISCLAIMER: If a successful answer is generated, always append a brief medical disclaimer at the very end starting from uppercased "DISCLAIMER: ".
        </instructions>

        <process>
        You must execute your task in two strict steps:

        Step 1: Evaluate the data inside <thinking> tags. 
        - Cross-reference the patient_query with the context. 
        - Determine if you have enough facts to form a safe, grounded response.
        - End your <thinking> block with exactly one of two tokens: [VALID] or [INVALID].

        Step 2: Generate the output.
        - If your conclusion was [INVALID], your final output outside the thinking tags MUST be exactly the escape phrase: [INSUFFICIENT_DATA]
        - If your conclusion was [VALID], output your fully cited, empathetic clinical response inside <final_answer> tags.
        </process>
        """
    
    print("🧠 Generating grounded response via Self-RAG...")
    # response = ollama.chat(
    #     model='llama3.1:8b', 
    #     messages=[{'role': 'user', 'content': prompt}],
    #     options={'temperature': 0.0}
    # )
    try:
        # Wywołanie API Groq
        # response = client.chat.completions.create(
        #     model="llama-3.3-70b-versatile", 
        #     messages=[{'role': 'user', 'content': prompt}],
        #     temperature=0.0 # W systemach QA wciąż potrzebujemy determinizmu
        # )
        
        # raw_answer = response.choices[0].message.content

        response = client.chat.completions.create(
            # Zmiana modelu na darmowy wariant z OpenRouter
            model= "meta-llama/llama-3.3-70b-instruct",#"meta-llama/llama-3.1-8b-instruct:free", #"meta-llama/llama-3.3-70b-instruct:free", 
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0 # W systemach QA wciąż potrzebujemy determinizmu
            # Opcjonalnie: dla OpenRouter warto dodać nagłówki referencyjne
            # extra_headers={
            #    "HTTP-Referer": "https://twojadomena.pl",
            #    "X-Title": "MedicalBot",
            # }
        )
        
        raw_answer = response.choices[0].message.content


        final_clean_answer = enforce_modern_guardrails(raw_answer)
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raw_answer = f"Error details: {e}"
        final_clean_answer = "⚠️ System Error: Failed to communicate with the API."

    
    # NEW: Return a structured dictionary for RAGAS and App integration
    return {
        "question": user_query,
        "answer": final_clean_answer,
        "contexts": [chunk['text'] for chunk in retrieved_chunks],
        "raw_llm_output": raw_answer # Helpful for debugging
    }

def enforce_modern_guardrails(raw_answer):
    """
    Modern XML parser matching the latest prompt architecture.
    """
    print(f"\n--- [DEBUG: System Logs] ---\n{raw_answer}\n---------------------------\n")
    
    # 1. Check for strict refusal
    if "[INSUFFICIENT_DATA]" in raw_answer:
        return "There is insufficient clinical data in the database to fully answer this specific question."
        
    # 2. Extract final patient answer
    final_answer_match = re.search(r"<final_answer>(.*?)</final_answer>", raw_answer, re.IGNORECASE | re.DOTALL)
    if final_answer_match:
        return final_answer_match.group(1).strip()
        
    return "⚠️ System Error: The model's response failed to meet strict safety formatting requirements."