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
    context = retrieve_context(conversation_str, k=3)
    
    system_prompt = f"""You are an expert AI cardiologist conducting an interactive assessment. 
Instead of writing a long essay immediately, act as a conversational doctor. 
If you need more information to make an accurate prediction, ask 1 or 2 short, direct follow-up questions (e.g., 'Do you have high blood pressure?', 'How long does the pain last?'). 
Once you have enough information, provide a SHORT, precise diagnostic note. 
When giving a diagnosis, you MUST include clear percentage probabilities for your top predictions (e.g., Coronary Artery Disease: 85%, Panic Attack: 15%).
Based ONLY on the following clinical medical context from our database, do not hallucinate outside facts. Keep your answers brief and conversational.

Medical Context:
{context}"""

    # Build the LLM messages payload
    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        # Strip the RAG proof from previous assistant messages to save context window
        content = msg["content"].split("\n\n---\n**📚 Proof of RAG")[0]
        llm_messages.append({"role": msg["role"], "content": content})
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=llm_messages,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Append the retrieved context to the response to prove it's using the dataset
        proof = f"\n\n---\n**📚 Proof of RAG (Context Retrieved from Database):**\n```text\n{context}\n```"
        
        return ai_response + proof
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
