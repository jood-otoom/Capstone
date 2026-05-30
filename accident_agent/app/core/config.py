# accident_agent/app/core/config.py

import os
from dotenv import load_dotenv

# 1. Resolve paths dynamically based on this file's location
# This steps backwards from app/core/config.py to the root folders
AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(AGENT_ROOT)

# 2. Point explicitly to the Single Source of Truth
root_env_path = os.path.join(PROJECT_ROOT, ".env")

# We keep override=False so that if ui_app/agent_service.py rotates the key 
# in os.environ, this load_dotenv doesn't accidentally overwrite it back to Key 1!
if os.path.exists(root_env_path):
    load_dotenv(dotenv_path=root_env_path, override=False)


class Settings:
    # Populated dynamically by ui_app, or falls back to the root .env
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Model configs
    VLM_MODEL = "google/gemini-2.5-flash"
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    VECTORSTORE_PATH = os.path.join(AGENT_ROOT, "data", "vectorstore")
    DOCS_DIR         = os.path.join(AGENT_ROOT, "docs")

settings = Settings()