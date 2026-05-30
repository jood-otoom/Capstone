# modelling/ui_app/agent_service.py

import os
import sys
from pathlib import Path
from ui_app.config import PROJECT_ROOT, AGENT_DIR
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

class APIKeyManager:
    def __init__(self):
        # Try reading keys from environment first (Docker / .env in project root).
        # Falls back to the hardcoded strings so local dev needs zero changes.
        env_keys = [
            os.getenv("OPENROUTER_API_KEY_1", "sk-or-v1"),
            os.getenv("OPENROUTER_API_KEY_2", "sk-or-v1"),
            os.getenv("OPENROUTER_API_KEY_3", "sk-or-v1"),
            os.getenv("OPENROUTER_API_KEY_4", "sk-or-v1"),
            os.getenv("OPENROUTER_API_KEY_5", "sk-or-v1"),
            os.getenv("OPENROUTER_API_KEY_6", "sk-or-v1"),   # 6th slot — no hardcoded fallback
        ]
        # Drop empty or unfilled placeholder values
        self.keys = [k for k in env_keys if k and not k.startswith("sk-or-v1-your")]
        if not self.keys:
            raise RuntimeError("No OpenRouter API keys found. Check OPENROUTER_API_KEY_1..6 in .env")
        self.current_index = 0
        print(f"[APIKeyManager] Loaded {len(self.keys)} OpenRouter keys.")

    def get_current_key(self) -> str:
        return self.keys[self.current_index]

    def rotate_key(self) -> str:
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.keys[self.current_index]
        os.environ["OPENROUTER_API_KEY"] = new_key
        # Also push into the agent's inner config so it takes effect immediately
        _push_key_to_agent_config(new_key)
        print(f"[APIKeyManager] Credit exhausted. Rotated to key index: {self.current_index} / {len(self.keys) - 1}")
        return new_key


def _push_key_to_agent_config(key: str) -> None:
    """Push a rotated key into accident_agent/app/core/config.py settings object."""
    try:
        from accident_agent.app.core.config import settings
        settings.OPENROUTER_API_KEY = key
    except Exception:
        pass


api_key_manager = APIKeyManager()

# Set the initial key in the environment so both the UI layer and the
# accident_agent layer start with the same key.
os.environ["OPENROUTER_API_KEY"] = api_key_manager.get_current_key()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "hf_your-token-here")  # Ensure HF_TOKEN is also set for the agent


def safe_agent_call(agent, method_name, *args, **kwargs):
    last_err = None
    # Track the current agent locally so we can swap it out if needed
    current_agent = agent 
    
    for attempt in range(len(api_key_manager.keys)):
        current_key = api_key_manager.get_current_key()
        os.environ["OPENROUTER_API_KEY"] = current_key
        _push_key_to_agent_config(current_key)

        try:
            # Call the method on the active agent
            method = getattr(current_agent, method_name)
            return method(*args, **kwargs)
            
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            print(f"[safe_agent_call] Attempt {attempt + 1} / key index "
                  f"{api_key_manager.current_index} failed: {e}")
                  
            is_credit_or_auth = any(word in err_msg for word in [
                "credit", "balance", "insufficient", "payment",
                "402", "unauthorized", "api_key", "401", "403",
            ])
            
            if is_credit_or_auth:
                # Rotate the key
                api_key_manager.rotate_key()
                # CRITICAL FIX: Re-instantiate the entire agent. 
                # This forces LangChain to drop the old cached client and build a new one.
                print("[safe_agent_call] Rebuilding agent with new API key...")
                current_agent = get_accident_agent()
            else:
                raise e
                
    raise RuntimeError(
        f"All {len(api_key_manager.keys)} API keys exhausted. Last error: {last_err}"
    )


def get_accident_agent():
    agent_path = PROJECT_ROOT / "accident_agent"
    if not agent_path.exists():
        raise FileNotFoundError(f"Accident Agent folder missing at: {agent_path}")

    for rf in ["app/services/agent_service.py", "app/core/graph_logic.py", "app/core/prompts.py"]:
        if not (agent_path / rf).exists():
            raise FileNotFoundError(f"Accident Agent missing file: {rf}")

    # Make sure the current key is in the environment before the agent reads it
    os.environ["OPENROUTER_API_KEY"] = api_key_manager.get_current_key()
    _push_key_to_agent_config(api_key_manager.get_current_key())

    from accident_agent.app.services.agent_service import AccidentAgent
    return AccidentAgent()
