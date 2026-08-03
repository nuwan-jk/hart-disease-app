import os
from rag import retrieve_context

def router_agent(user_input: str) -> str:
    """
    Router Agent: Uses simple heuristics (or a fast LLM) to classify intent.
    Intent categories: 'greeting', 'medical_symptom', 'other'
    """
    input_lower = user_input.lower()
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'test']
    
    if any(greet in input_lower for greet in greetings):
        return "greeting"
    elif any(word in input_lower for word in ['pain', 'chest', 'breath', 'sweat', 'fatigue', 'heart', 'dizzy', 'numb']):
        return "medical_symptom"
    else:
        return "other"

def medical_synthesizer_agent(user_input: str) -> str:
    """
    Synthesizer Agent: Retrieves context and generates a diagnosis prediction.
    """
    print("[Agent] Retrieving medical context from vector DB...")
    context = retrieve_context(user_input)
    
    print("[Agent] Synthesizing response using High Reasoning Model...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return f"**(Mock Response - OpenRouter API Key missing)**\n\nBased on your symptoms and our medical database, here is a prediction.\n\n**Retrieved Context:**\n{context}\n\n*Please provide OpenRouter API key in .env for real AI inference.*"
    
    # In a fully connected version, this would call OpenRouter API using LangChain/OpenAI client
    return f"**(Real Inference Mode)**\nRetrieved Context:\n{context}\n\nPrediction: (API connection code would run here)"

def orchestrator(user_input: str) -> str:
    """
    Orchestrator: Routes the user input to the correct agent.
    """
    intent = router_agent(user_input)
    if intent == "greeting":
        return "Hello! I am a Heart Disease Prediction AI. Please describe your symptoms (e.g., chest pain, shortness of breath) and I will try to assess your condition."
    elif intent == "medical_symptom":
        return medical_synthesizer_agent(user_input)
    else:
        return "I am specialized in heart disease prediction. Please describe any heart-related symptoms you are experiencing."

if __name__ == "__main__":
    print("Testing Orchestrator...")
    print(orchestrator("Hello"))
    print("-" * 20)
    print(orchestrator("I have severe chest pain and sweating."))
