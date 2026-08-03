# 🫀 Agentic AI Heart Disease Predictor

**Live Demo URL:** [https://hart-disease-app.streamlit.app/](https://hart-disease-app.streamlit.app/)

## 👨‍🎓 Student Information
- **Name:** IWJGN Jaya Keerthisinghe
- **Student ID:** ITBIN-2313-0050

## 📖 Project Overview
This project is an **Agentic AI Application** developed for predicting the likelihood of heart disease based on user symptoms. It employs a **multi-agent architecture** combined with a **Retrieval-Augmented Generation (RAG)** pipeline to simulate an interactive medical consultation.

Unlike standard LLMs that generate immediate long-form text, this system acts progressively. It acts as an interactive doctor, asking follow-up questions to gather at least 3 clinical data points before providing a final diagnostic percentage.

## ⚙️ Core Technologies & Architecture
1. **Frontend UI:** Built using **Streamlit**, providing a clean, conversational chat interface where the user communicates with the medical AI.
2. **Agentic Routing:** Utilizes **Groq (Llama-3.1-8B-Instant)** to classify user intent (greeting vs. medical symptom).
3. **Medical Synthesizer:** Utilizes **Groq (Llama-3.3-70B-Versatile)** for high-level deep reasoning and diagnostic predictions based on RAG.
4. **RAG Pipeline (Knowledge Base):**
   - Implemented using **LangChain** and **ChromaDB**.
   - Contains 20 highly professional, custom-generated Medical Clinical Guidelines.
   - Embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`.

## 🌟 Key Features
- **Strict Phase Workflow:** The AI is strictly prompted to gather sufficient symptom information (asking questions one by one) *before* diagnosing.
- **RAG Data Provenance:** The application transparently displays the exact retrieved context from the ChromaDB database via a neat UI expander, proving that the model is grounding its answers in the provided clinical data rather than hallucinating.
- **Percentage Probabilities:** Provides concise final notes with percentage probabilities for specific conditions.

## 🚀 How to Run Locally

1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with your API keys:
   ```env
   GROQ_API_KEY="your-groq-key"
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
