import ollama
import os
import json
import re
from groq import Groq 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
groq_key = os.environ.get("GROQ_API_KEY")
# client = Groq(api_key=groq_key)

# Inicjalizacja klienta wskazująca na OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

def decompose_query(user_query):
    """
    Przyjmuje zapytanie pacjenta i rozbija je na atomowe pod-zapytania.
    Zwraca listę stringów w Pythonie.
    """
    # prompt = f"""
    #     <role>
    #     You are an expert Query Decomposer for a Medical RAG system.
    #     Your only task is to break down complex patient queries into distinct, atomic search queries.
    #     </role>

    #     <instructions>
    #     1. Analyze the user's query. If it contains multiple distinct medical concepts, questions, or entities, split it.
    #     2. If the query is already simple and asks about only one thing, return an array with just that one query.
    #     3. OUTPUT FORMAT: You MUST output ONLY a valid JSON array of strings. Do not write ANY conversational text before or after the JSON.
    #     4. Keep the atomic queries in the original language of the user (English).
    #     </instructions>

    #     <example_input>
    #     What are the side effects of aspirin and what is the safe dosage and how much does an MRI cost?
    #     </example_input>
    #     <example_output>
    #     ["What are the side effects of aspirin?", "What is the safe dosage of aspirin?", "How much does an MRI cost?"]
    #     </example_output>

    #     <user_query>
    #     {user_query}
    #     </user_query>
    #     """
    prompt = f"""
        <role>
        You are an expert Clinical Intent Extractor for a Medical RAG system.
        Your ONLY task is to extract the core medical questions from a patient's message, ignoring all emotional context, greetings, and irrelevant personal backstory.
        </role>

        <instructions>
        1. Analyze the message and identify the core medical knowledge needed to assist the patient.
        2. Formulate a MAXIMUM of 2 to 3 concise, highly specific medical search queries. 
        3. Discard noise (e.g., "I am from India", "Please help me", "Contact me").
        4. Retain critical clinical context in the queries (e.g., age, specific disease, stage).
        5. OUTPUT FORMAT: You MUST output ONLY a valid JSON array of strings. Do not write ANY conversational text before or after the JSON.
        </instructions>

        <example_input>
        Dear Sir. I am from India. My father is 61 years old suffering with Pancreatic cancer and it's in 3/4 Rd stage. All doctors are helpless. I want to save his life. Please suggest treatments.
        </example_input>
        <example_output>
        ["What are the treatment options for stage 3 and 4 pancreatic cancer?", "What is the prognosis and palliative care for a 61-year-old with advanced pancreatic cancer?"]
        </example_output>

        <user_query>
        {user_query}
        </user_query>
        """
    
    print("🔪 [ORCHESTRATOR] Analiza i dekompozycja zapytania...")
    try:
        # response = ollama.chat(
        #     model='llama3.1:8b', 
        #     messages=[{'role': 'user', 'content': prompt}],
        #     options={'temperature': 0.0}# Absolutny determinizm
        # )
        # raw_output = response['message']['content']

        # response = client.chat.completions.create(
        #     model="llama-3.1-8b-instant", 
        #     messages=[{'role': 'user', 'content': prompt}],
        #     temperature=0.0 # <-- ZMIANA TUTAJ (brak słowa 'options' i klamerek)
        # )
        response = client.chat.completions.create(
            model= "meta-llama/llama-3.1-8b-instruct", #"meta-llama/llama-3.1-8b-instruct:free", #"meta-llama/llama-3.3-70b-instruct:free", 
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0)
            
        
        raw_output = response.choices[0].message.content
        
        # Ekstrakcja czystego JSON-a (usuwa znaki formatowania markdown typu ```json)
        json_match = re.search(r"\[.*\]", raw_output, re.DOTALL)
        
        if json_match:
            clean_json = json_match.group(0)
            sub_queries = json.loads(clean_json) # Zamiana stringa na listę w Pythonie
            return sub_queries
        else:
            print("⚠️ Nie udało się wyciągnąć JSONa. Używam oryginalnego zapytania.")
            return [user_query]
            
    except Exception as e:
        print(f"❌ Błąd dekompozycji: {e}")
        return [user_query] # Bezpiecznik: jeśli LLM padnie, zwracamy oryginał jako 1-elementową listę