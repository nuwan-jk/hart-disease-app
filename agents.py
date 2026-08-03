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
except Exception as e:
    print(f"Error initializing API clients: {e}")

def router_agent(user_input: str) -> str:
    """
    Router Agent: Uses Llama 3.1 8B on Groq to classify intent.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
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

def medical_synthesizer_agent(messages: list) -> str:
    """
    Synthesizer Agent: Retrieves context and generates a conversational diagnosis prediction using Groq.
    """
    latest_user_input = messages[-1]["content"]
    
    # Create a string representation of the conversation for context retrieval
    conversation_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    raw_context = retrieve_context(conversation_str, k=3)
    
    # Deduplicate the context chunks to prevent ugly repetitive text
    unique_chunks = []
    for chunk in raw_context.split('\n\n'):
        if chunk.strip() and chunk.strip() not in unique_chunks:
            unique_chunks.append(chunk.strip())
    context = "\n\n".join(unique_chunks)
    
    system_prompt = f"""You are an expert AI cardiologist conducting an interactive assessment. 
Follow this STRICT workflow:

PHASE 1: GATHER INFORMATION
- You MUST ask the user relevant follow-up questions ONE AT A TIME based on their symptoms.
- Continue asking questions (one per response) until you have collected at least 3 distinct pieces of medical information from the user's answers. 
- DO NOT provide any diagnosis, summary, or percentages during Phase 1. Just ask the next question and wait for the user to answer.

PHASE 2: DIAGNOSIS
- ONLY when you have gathered enough information (at least 3 responses), stop asking questions.
- Provide a SHORT, precise diagnostic note. 
- You MUST include clear percentage probabilities for your top predictions (e.g., Coronary Artery Disease: 85%, Normal: 15%).
- Based ONLY on the following clinical medical context from our database, do not hallucinate outside facts.

Medical Context:
{context}"""

    # Build the LLM messages payload
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        # Strip the RAG context from previous assistant messages to save context window
        content = msg["content"].split("|||CONTEXT|||")[0].strip()
        llm_messages.append({"role": msg["role"], "content": content})
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=llm_messages,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Append the retrieved context using a hidden delimiter so UI can parse it
        return f"{ai_response}|||CONTEXT|||{context}"
    except Exception as e:
        return f"Error communicating with AI Synthesizer: {e}"

def orchestrator(messages: list) -> str:
    """
    Orchestrator: Routes the user input to the correct agent.
    """
    latest_user_input = messages[-1]["content"]
    intent = router_agent(latest_user_input)
    
    if intent == "greeting":
        return "Hello! I am a Heart Disease Prediction AI. Please describe your symptoms (e.g., chest pain, shortness of breath) and I will assess your condition based on our clinical guidelines."
    elif intent == "medical_symptom":
        return medical_synthesizer_agent(messages)
    else:
        return "I am specialized in heart disease prediction. Please describe any heart-related symptoms you are experiencing."

if __name__ == "__main__":
    # Test with a dummy message list
    dummy_messages = [{"role": "user", "content": "I have chest pain and shortness of breath."}]
    print(orchestrator(dummy_messages))
