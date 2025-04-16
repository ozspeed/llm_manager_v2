import os
import json
import logging
import traceback
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_SETTINGS = {
    "app": {
        "version": "1.2.0",
        "port": 8001,
        "debug": False
    },
    "paths": {
        "model_library": "",
        "database": "models.db",
        "draft_download_area": ""
    },
    "models": {
        "extensions": [".gguf", ".ggml", ".bin", ".safetensors", ".onnx", ".pt", ".pth"]
    },
    "frameworks": {
        "ollama": {
            "enabled": False,
            "repositories": []
        },
        "huggingface": {
            "enabled": False,
            "api_token": "",
            "search_results_limit": 50,
            "trending_search": {
                "query": "",
                "model_type": "text-generation",
                "tags": ["gguf"],
                "sort_by": "last_modified",
                "sort_direction": -1
            }
        }
    },
    "tool_repositories": []
}

_settings = None

SETTINGS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models.db')
SETTINGS_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')

# =============================================================================
# DATABASE-DRIVEN SETTINGS MANAGEMENT
# =============================================================================

def init_settings_db():
    """
    Create the settings table if it does not exist, enforcing UNIQUE(section, key) constraint for upsert support.
    """
    try:
        with sqlite3.connect(SETTINGS_DB_PATH) as conn:
            # Try to add the unique constraint if not present
            try:
                conn.execute('''CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    section TEXT,
                    key TEXT,
                    value TEXT,
                    UNIQUE(section, key)
                )''')
            except Exception as e:
                logger.warning(f"Table already exists, checking UNIQUE constraint: {e}")
            conn.commit()
    except Exception as e:
        logger.error(f"Error initializing settings DB: {e}")

def get_db_settings() -> Dict[str, Any]:
    """
    Fetch all settings from the DB and reconstruct as a hierarchical dict.
    This ensures the UI/API receives settings in the expected nested format.
    """
    settings = {}
    try:
        with sqlite3.connect(SETTINGS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT section, key, value FROM settings')
            for section, key, value in cursor.fetchall():
                # Parse JSON values if possible
                try:
                    val = json.loads(value)
                except Exception:
                    val = value
                if section not in settings:
                    settings[section] = {}
                settings[section][key] = val
    except Exception as e:
        logger.error(f"Error loading settings from DB: {e}")
    return merge_with_defaults(settings)

def merge_with_defaults(loaded: Dict[str, Any]) -> Dict[str, Any]:
    """Merge loaded settings with defaults, filling in missing sections/keys."""
    def deep_merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_merge(d.get(k, {}), v)
            else:
                d.setdefault(k, v)
        return d
    return deep_merge(loaded, DEFAULT_SETTINGS.copy())

def set_db_setting(section: str, key: str, value: Any) -> bool:
    """Set or update a single setting in the DB."""
    try:
        value_str = json.dumps(value)
        with sqlite3.connect(SETTINGS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO settings (section, key, value) VALUES (?, ?, ?)
                ON CONFLICT(section, key) DO UPDATE SET value=excluded.value''', (section, key, value_str))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error setting DB setting: {e}")
        return False

def get_db_setting(section: str, key: str, default: Any = None) -> Any:
    try:
        with sqlite3.connect(SETTINGS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE section=? AND key=?', (section, key))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except Exception:
                    return row[0]
    except Exception as e:
        logger.error(f"Error getting DB setting: {e}")
    return default

def delete_db_setting(section: str, key: str) -> bool:
    try:
        with sqlite3.connect(SETTINGS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM settings WHERE section=? AND key=?', (section, key))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting DB setting: {e}")
        return False

def save_hierarchical_settings(hier: Dict[str, Any]):
    """Recursively save all settings from a hierarchical dict to the DB."""
    for section, keys in hier.items():
        if not isinstance(keys, dict):
            continue
        for key, value in keys.items():
            set_db_setting(section, key, value)

def migrate_json_to_db():
    """If DB is empty, migrate settings from JSON and back up JSON file."""
    if not os.path.exists(SETTINGS_JSON_PATH):
        return
    try:
        with sqlite3.connect(SETTINGS_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM settings')
            if cursor.fetchone()[0] > 0:
                return  # Already migrated
        with open(SETTINGS_JSON_PATH, 'r') as f:
            settings = json.load(f)
        # Flatten and save
        for section, keys in settings.items():
            if isinstance(keys, dict):
                for key, value in keys.items():
                    set_db_setting(section, key, value)
        # Backup JSON
        os.rename(SETTINGS_JSON_PATH, SETTINGS_JSON_PATH + '.bak')
        logger.info('Migrated settings.json to database and created backup.')
    except Exception as e:
        logger.error(f"Error migrating JSON to DB: {e}")

# =============================================================================
# SETTINGS MANAGEMENT - Core functions for loading and saving settings
# =============================================================================

def load_settings() -> None:
    """Load settings from database (preferred) or fallback to settings.json."""
    global _settings, config
    try:
        init_settings_db()
        migrate_json_to_db()
        _settings = get_db_settings()
        logger.info("Loaded settings from database.")
    except Exception as e:
        logger.error(f"Error loading settings from DB, falling back to JSON: {e}")
        # Fallback to JSON
        try:
            with open(SETTINGS_JSON_PATH, 'r') as f:
                _settings = json.load(f)
        except Exception as e2:
            logger.error(f"Error loading settings from JSON: {e2}")
            _settings = DEFAULT_SETTINGS.copy()
    _update_config_reference()

def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Recursively update target dict with values from source dict.
    
    Args:
        target: The target dictionary to update
        source: The source dictionary with values to apply
        
    Returns:
        None
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = value

def save_settings() -> bool:
    """Save current settings to the database."""
    try:
        if _settings is None:
            logger.error("No settings loaded to save.")
            return False
        save_hierarchical_settings(_settings)
        logger.info("Saved settings to database.")
        return True
    except Exception as e:
        logger.error(f"Error saving settings to DB: {e}")
        return False

# =============================================================================
# SETTINGS ACCESS - Functions for getting and setting values
# =============================================================================

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value by key using dot notation (DB-backed)."""
    if _settings is None:
        load_settings()
    try:
        value = _settings
        for k in key.split('.'):
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def update_settings(settings_dict: Dict[str, Any]) -> bool:
    """Update multiple settings at once (DB-backed)."""
    if _settings is None:
        load_settings()
    try:
        # Convert flat dict to nested dict and update in-memory
        for key, value in settings_dict.items():
            keys = key.split('.')
            d = _settings
            for k in keys[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        # Save to DB
        save_hierarchical_settings(_settings)
        return True
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return False

def set_setting(key: str, value: Any) -> bool:
    """Set a setting value by key using dot notation (DB-backed)."""
    if _settings is None:
        load_settings()
    try:
        keys = key.split('.')
        d = _settings
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        # Save to DB
        save_hierarchical_settings(_settings)
        target[keys[-1]] = value
        
        # Save settings to file
        save_settings()
        return True
    except Exception as e:
        logger.error(f"Error setting configuration value: {str(e)}")
        return False

# =============================================================================
# COMMON GETTERS - Convenience functions for accessing specific settings
# =============================================================================

def get_model_library_path() -> str:
    """Get the path to the model library.
    
    Returns:
        str: The expanded path to the model library
    """
    return os.path.expanduser(get_setting('paths.model_library', DEFAULT_SETTINGS['paths']['model_library']))

def get_database_path() -> str:
    """Get the path to the database file.
    
    Returns:
        str: The path to the database file
    """
    return get_setting('paths.database', DEFAULT_SETTINGS['paths']['database'])

def get_draft_download_area() -> str:
    """Get the path to the draft download area.
    
    Returns:
        str: The expanded path to the draft download area
    """
    return os.path.expanduser(get_setting('paths.draft_download_area', DEFAULT_SETTINGS['paths']['draft_download_area'] or '~/Downloads/LLM_Downloads'))

def get_model_extensions() -> List[str]:
    """Get the list of supported model file extensions.
    
    Returns:
        List[str]: List of supported model file extensions
    """
    return get_setting('models.extensions', DEFAULT_SETTINGS['models']['extensions'])

def get_port() -> int:
    """Get the port number for the application server.
    
    Returns:
        int: The port number
    """
    return get_setting('app.port', DEFAULT_SETTINGS['app']['port'])

def get_debug_mode() -> bool:
    """Get whether debug mode is enabled.
    
    Returns:
        bool: True if debug mode is enabled, False otherwise
    """
    return get_setting('app.debug', DEFAULT_SETTINGS['app']['debug'])

def get_ollama_enabled() -> bool:
    """Get whether Ollama integration is enabled.
    
    Returns:
        bool: True if Ollama integration is enabled, False otherwise
    """
    return get_setting('frameworks.ollama.enabled', DEFAULT_SETTINGS['frameworks']['ollama']['enabled'])

def get_ollama_repositories() -> List[str]:
    """Get the list of Ollama repositories.
    
    Returns:
        List[str]: List of Ollama repositories
    """
    return get_setting('frameworks.ollama.repositories', DEFAULT_SETTINGS['frameworks']['ollama']['repositories'])

def get_huggingface_enabled() -> bool:
    """Get whether Hugging Face integration is enabled.
    
    Returns:
        bool: True if Hugging Face integration is enabled, False otherwise
    """
    return get_setting('frameworks.huggingface.enabled', DEFAULT_SETTINGS['frameworks']['huggingface']['enabled'])

def get_huggingface_api_token() -> str:
    """Get the Hugging Face API token.
    
    Returns:
        str: The Hugging Face API token
    """
    return get_setting('frameworks.huggingface.api_token', DEFAULT_SETTINGS['frameworks']['huggingface']['api_token'])

def get_app_version() -> str:
    """Get the application version.
    
    Returns:
        str: The application version
    """
    return get_setting('app.version', DEFAULT_SETTINGS['app']['version'])

def get_tool_repositories() -> List[Dict[str, Any]]:
    """Get the list of tool repositories.
    
    Returns:
        List[Dict[str, Any]]: List of tool repositories
    """
    return get_setting('tool_repositories', DEFAULT_SETTINGS['tool_repositories'])

# =============================================================================
# COMPATIBILITY FUNCTIONS - For backward compatibility with older code
# =============================================================================

def load_config() -> None:
    """Alias for load_settings for backward compatibility.
    
    Returns:
        None
    """
    load_settings()

def save_config() -> bool:
    """Alias for save_settings for backward compatibility.
    
    Returns:
        bool: True if settings were saved successfully, False otherwise
    """
    return save_settings()

# Make the settings accessible as config.config to maintain compatibility
# Create a config variable that points to _settings
config = _settings

# Update the config reference whenever _settings is updated
def _update_config_reference() -> None:
    """Update the config reference to point to the current _settings.
    
    This function is used to maintain compatibility with code that uses the
    config.config pattern to access settings.
    
    Returns:
        None
    """
    global config, _settings
    config = _settings

# Load settings on module import
load_settings()
