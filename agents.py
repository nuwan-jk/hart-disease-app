import os
from openai import OpenAI
from rag import retrieve_context
from dotenv import load_dotenv

load_dotenv()

# Initialize clients using OpenAI compatible endpoints
try:
    groq_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.environ.get("GROQ_API_KEY")
    )
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )
except Exception as e:
    print(f"Error initializing API clients: {e}")

def router_agent(user_input: str) -> str:
    """
    Router Agent: Uses Llama 3 8B on Groq to classify intent.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an intent router. Classify the user input into exactly one of these three categories: 'greeting', 'medical_symptom', 'other'. Respond with ONLY the category name and nothing else."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0
        )
        intent = response.choices[0].message.content.strip().lower()
        if 'greeting' in intent: return 'greeting'
        if 'medical' in intent: return 'medical_symptom'
        if 'other' in intent: return 'other'
        return "other"
    except Exception as e:
        print(f"Router error: {e}")
        return "medical_symptom" # fallback

def medical_synthesizer_agent(user_input: str) -> str:
    """
    Synthesizer Agent: Retrieves context and generates a diagnosis prediction using OpenRouter.
    """
    context = retrieve_context(user_input, k=3)
    
    prompt = f"""You are an expert AI cardiologist. Based ONLY on the following clinical medical context from our database and the user's symptoms, provide a potential assessment. Include a percentage likelihood if possible, and clearly state that this is an AI prediction, not a replacement for a real doctor. Do not make up facts outside the context.

User Symptoms:
{user_input}

Medical Context:
{context}

Response:"""
    
    try:
        response = openrouter_client.chat.completions.create(
            model="google/gemma-2-9b-it:free",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with AI Synthesizer: {e}"

def orchestrator(user_input: str) -> str:
    """
    Orchestrator: Routes the user input to the correct agent.
    """
    intent = router_agent(user_input)
    if intent == "greeting":
        return "Hello! I am a Heart Disease Prediction AI. Please describe your symptoms (e.g., chest pain, shortness of breath) and I will assess your condition based on our clinical guidelines."
    elif intent == "medical_symptom":
        return medical_synthesizer_agent(user_input)
    else:
        return "I am specialized in heart disease prediction. Please describe any heart-related symptoms you are experiencing."

if __name__ == "__main__":
    print(orchestrator("I have chest pain and shortness of breath."))
