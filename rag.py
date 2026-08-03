import os
# Workaround for Streamlit Cloud SQLite version issue with ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = "chroma_db"
DATA_DIR = "data"

def get_vector_store():
    # Use a lightweight embedding model that runs locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("Loading existing vector database...")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return db
    
    print("Vector database not found or empty. Ingesting documents...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Chunking strategy: 500 characters with 50 character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
    print(f"Ingested {len(docs)} chunks into ChromaDB.")
    return db

def retrieve_context(query, k=3):
    db = get_vector_store()
    results = db.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context

if __name__ == "__main__":
    # Test the pipeline
    get_vector_store()
    print("\n--- Test Retrieval ---")
    print(retrieve_context("What are the symptoms of a heart attack?"))

# Ensure DB is persistent
