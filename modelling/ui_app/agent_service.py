# modelling/ui_app/agent_service.py
import os

from dotenv import load_dotenv

from ui_app.config import PROJECT_ROOT


load_dotenv(PROJECT_ROOT / ".env", override=True)


class APIKeyManager:
    def __init__(self):
        env_keys = []

        primary_key = os.getenv("OPENROUTER_API_KEY")
        if primary_key:
            env_keys.append(primary_key)

        for idx in range(1, 7):
            env_keys.append(os.getenv(f"OPENROUTER_API_KEY_{idx}", ""))

        self.keys = [
            key
            for key in env_keys
            if key and not key.startswith("sk-or-v1-your") and key != "sk-or-v1"
        ]
        self.current_index = 0
        print(f"[APIKeyManager] Loaded {len(self.keys)} OpenRouter keys.")

    def has_keys(self) -> bool:
        return bool(self.keys)

    def get_current_key(self) -> str:
        if not self.keys:
            raise RuntimeError("No OpenRouter API keys found. Check OPENROUTER_API_KEY or OPENROUTER_API_KEY_1..6 in .env")
        return self.keys[self.current_index]

    def rotate_key(self) -> str:
        if not self.keys:
            raise RuntimeError("Cannot rotate OpenRouter API key because no keys are configured.")
        self.current_index = (self.current_index + 1) % len(self.keys)
        new_key = self.keys[self.current_index]
        os.environ["OPENROUTER_API_KEY"] = new_key
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

if api_key_manager.has_keys():
    os.environ["OPENROUTER_API_KEY"] = api_key_manager.get_current_key()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token


def safe_agent_call(agent, method_name, *args, **kwargs):
    if not api_key_manager.has_keys():
        raise RuntimeError("No OpenRouter API keys configured for the accident agent.")

    last_err = None
    current_agent = agent

    for attempt in range(len(api_key_manager.keys)):
        current_key = api_key_manager.get_current_key()
        os.environ["OPENROUTER_API_KEY"] = current_key
        _push_key_to_agent_config(current_key)

        try:
            method = getattr(current_agent, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_msg = str(e).lower()
            print(
                f"[safe_agent_call] Attempt {attempt + 1} / key index "
                f"{api_key_manager.current_index} failed: {e}"
            )

            is_credit_or_auth = any(
                word in err_msg
                for word in [
                    "credit",
                    "balance",
                    "insufficient",
                    "payment",
                    "402",
                    "unauthorized",
                    "api_key",
                    "401",
                    "403",
                ]
            )

            if is_credit_or_auth:
                api_key_manager.rotate_key()
                print("[safe_agent_call] Rebuilding agent with new API key...")
                current_agent = get_accident_agent()
            else:
                raise

    raise RuntimeError(
        f"All {len(api_key_manager.keys)} API keys exhausted. Last error: {last_err}"
    )


def get_accident_agent():
    agent_path = PROJECT_ROOT / "accident_agent"
    if not agent_path.exists():
        raise FileNotFoundError(f"Accident Agent folder missing at: {agent_path}")

    for required_file in [
        "app/services/agent_service.py",
        "app/core/graph_logic.py",
        "app/core/prompts.py",
    ]:
        if not (agent_path / required_file).exists():
            raise FileNotFoundError(f"Accident Agent missing file: {required_file}")

    if api_key_manager.has_keys():
        current_key = api_key_manager.get_current_key()
        os.environ["OPENROUTER_API_KEY"] = current_key
        _push_key_to_agent_config(current_key)

    from accident_agent.app.services.agent_service import AccidentAgent

    return AccidentAgent()
