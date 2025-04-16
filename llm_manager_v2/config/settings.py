import os
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env', override=False)

def get_env(key: str, default: Any = None, cast_type: Optional[type] = None) -> Any:
    value = os.environ.get(key, default)
    if cast_type and value is not None:
        try:
            return cast_type(value)
        except Exception:
            return default
    return value

class Config:
    """App configuration loader for v2."""
    # Example settings
    DEBUG: bool = bool(get_env('DEBUG', False, cast_type=bool))
    SECRET_KEY: str = get_env('SECRET_KEY', 'changeme')
    DB_PATH: str = get_env('DB_PATH', 'models.db')
    # Add more settings as needed

    @classmethod
    def from_db(cls) -> 'Config':
        # TODO: Load config from SQLite DB (stub)
        # Example: fetch settings from settings table and override env values
        return cls()

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        return {k: getattr(cls, k) for k in dir(cls) if k.isupper()}

def get_config() -> Config:
    # In future: merge env, .env, and DB config
    return Config.from_db()
