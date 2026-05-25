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
    
    # Paths
    VECTORSTORE_PATH = "data/vectorstore"
    DOCS_DIR = "docs" # Changed from a single file to a directory

settings = Settings()