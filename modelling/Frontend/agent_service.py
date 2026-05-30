import os
import sys
from pathlib import Path
from ui_app.config import PROJECT_ROOT, AGENT_DIR

class APIKeyManager:
    def __init__(self):
        self.keys = [
            "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",
            "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"
        ]
        self.current_index = 0

    def get_current_key(self) -> str:
        return self.keys[self.current_index]

    def rotate_key(self) -> str:
        self.current_index = (self.current_index + 1) % len(self.keys)
        os.environ["OPENROUTER_API_KEY"] = self.keys[self.current_index]
        try:
            from app.core.config import settings
            settings.OPENROUTER_API_KEY = self.keys[self.current_index]
        except Exception:
            pass
        print(f"[APIKeyManager] Credit exhausted or API key failed. Rotated to key index: {self.current_index}")
        return self.keys[self.current_index]
api_key_manager = APIKeyManager()
os.environ['OPENROUTER_API_KEY'] = api_key_manager.get_current_key()
os.environ['HF_TOKEN'] = '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'

def safe_agent_call(agent, method_name, *args, **kwargs):
    last_err = None
    for attempt in range(len(api_key_manager.keys)):
        current_key = api_key_manager.get_current_key()
        os.environ["OPENROUTER_API_KEY"] = current_key
        try:
            from app.core.config import settings
            settings.OPENROUTER_API_KEY = current_key
        except Exception:
            pass
            
        # Update LLM dynamically if it exists
        if agent and hasattr(agent, "llm") and agent.llm:
            if hasattr(agent.llm, "openai_api_key"):
                agent.llm.openai_api_key = current_key
            if hasattr(agent.llm, "api_key"):
                agent.llm.api_key = current_key

        try:
            method = getattr(agent, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            print(f"[safe_agent_call] Attempt {attempt+1} using key index {api_key_manager.current_index} failed: {e}")
            
            # Rotate key on credit or auth failures
            is_credit_or_auth = any(word in err_msg for word in [
                "credit", "balance", "insufficient", "payment", "402", "unauthorized", "api_key", "401", "403"
            ])
            if is_credit_or_auth:
                api_key_manager.rotate_key()
            else:
                raise e
    raise RuntimeError(f"All API keys failed or exhausted credit. Last error: {last_err}")

def get_accident_agent():
    """
    Safely load and instantiate the AccidentAgent.
    Returns the agent instance or raises descriptive errors.
    """
    agent_path = PROJECT_ROOT / "accident_agent"
    if not agent_path.exists():
        raise FileNotFoundError(f"Accident Agent folder is missing at: {agent_path}")

    required_files = ["app/services/agent_service.py", "app/core/graph_logic.py", "app/core/prompts.py"]
    for rf in required_files:
        p = agent_path / rf
        if not p.exists():
            raise FileNotFoundError(f"Accident Agent is missing crucial file: {rf}")

    os.environ["OPENROUTER_API_KEY"] = api_key_manager.get_current_key()
    
    from app.services.agent_service import AccidentAgent
    return AccidentAgent()
