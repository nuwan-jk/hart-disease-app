import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We import the orchestrator from agents.py
try:
    from agents import orchestrator
except Exception as e:
    import traceback
    err_msg = traceback.format_exc()
    # Fallback if agents.py has missing dependencies on Streamlit Cloud
    def orchestrator(user_input):
        return f"**System Error**: Agents module failed to load. Please check `requirements.txt`.\n\nDetails:\n```text\n{err_msg}\n```"

st.set_page_config(page_title="Heart Disease AI Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Prediction AI")
st.markdown("""
Welcome to the Agentic Heart Disease Predictor.
This app uses a combination of RAG (Retrieval-Augmented Generation) and Agentic reasoning to assess your symptoms based on a medical knowledge base.
""")

# Check for API Keys
openrouter_key = os.getenv("OPENROUTER_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

if not openrouter_key or not groq_key:
    st.warning("⚠️ **API Keys Missing**: Please make sure `OPENROUTER_API_KEY` and `GROQ_API_KEY` are set in your `.env` file for real AI predictions.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "|||CONTEXT|||" in message["content"]:
            resp, ctx = message["content"].split("|||CONTEXT|||")
            st.markdown(resp)
            if ctx.strip():
                with st.expander("📚 View RAG Context"):
                    st.markdown(f"```text\n{ctx.strip()}\n```")
        else:
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Describe your symptoms here... (e.g., 'I have chest pain and shortness of breath')"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call our agent orchestrator
    with st.spinner("Analyzing symptoms and retrieving medical context..."):
        full_response = orchestrator(st.session_state.messages)
    
    if "|||CONTEXT|||" in full_response:
        response_text, context_text = full_response.split("|||CONTEXT|||")
    else:
        response_text = full_response
        context_text = ""
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if context_text.strip():
            with st.expander("📚 View RAG Context"):
                st.markdown(f"```text\n{context_text.strip()}\n```")
                
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Session state persistence

# API keys validation

# Final UI polish
