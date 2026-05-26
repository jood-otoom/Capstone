# app/core/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # Model configs
    VLM_MODEL = "google/gemini-2.5-flash" 
    # Upgraded to support both Arabic and English semantic search
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
    
    # Resolve dynamic agent root path relative to this file
    AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Paths
    VECTORSTORE_PATH = os.path.join(AGENT_ROOT, "data", "vectorstore")
    DOCS_DIR = os.path.join(AGENT_ROOT, "docs") # Changed from a single file to a directory

settings = Settings()