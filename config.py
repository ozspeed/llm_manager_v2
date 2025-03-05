"""
Configuration settings for the LLM Model Manager application.
This module provides a configuration system that can load and save settings to a JSON file.
"""

import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('config')

# Application version
APP_VERSION = "1.1.0"

# Configuration file path
CONFIG_FILE = "settings.json"

# Default configuration
DEFAULT_CONFIG = {
    "app": {
        "version": APP_VERSION,
        "port": 8001,
        "debug": True
    },
    "paths": {
        "model_library": "/Volumes/Library_Bolt/AI Model Library",
        "database": "models.db"
    },
    "models": {
        "extensions": [".gguf", ".ggml", ".bin", ".safetensors", ".onnx", ".pt", ".pth"]
    },
    "frameworks": {
        "ollama": {
            "enabled": True,
            "repositories": ["ollama"]
        }
    },
    "tool_repositories": [
        {
            "name": "Ollama",
            "use_ai_library": True,  # Kept for backward compatibility
            "symlink_from_library": True,
            "repository_location": "/usr/local/bin/ollama",
            "require_parent_directory": False,
            "understands_shards": False,
            "audit_trail": []
        },
        {
            "name": "GPT4All",
            "use_ai_library": False,  # Kept for backward compatibility
            "symlink_from_library": False,
            "repository_location": "",
            "require_parent_directory": False,
            "understands_shards": False,
            "audit_trail": []
        }
    ]
}

# Global configuration object
config = {}

def load_config():
    """Load configuration from file or create with defaults if it doesn't exist."""
    global config
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                loaded_config = json.load(f)
                logger.info(f"Loaded configuration from {CONFIG_FILE}")
                
                # Merge with defaults to ensure all keys exist
                config = DEFAULT_CONFIG.copy()
                update_nested_dict(config, loaded_config)
        else:
            config = DEFAULT_CONFIG.copy()
            save_config()
            logger.info(f"Created new configuration file at {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        config = DEFAULT_CONFIG.copy()
    
    return config

def save_config():
    """Save current configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration to {CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def update_nested_dict(d, u):
    """Recursively update a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            update_nested_dict(d[k], v)
        else:
            d[k] = v

def get_setting(path, default=None):
    """Get a setting value using dot notation path."""
    parts = path.split('.')
    current = config
    
    for part in parts:
        if part in current:
            current = current[part]
        else:
            return default
    
    return current

def set_setting(path, value):
    """Set a setting value using dot notation path."""
    parts = path.split('.')
    current = config
    
    # Navigate to the correct nested dictionary
    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Set the value
    current[parts[-1]] = value
    
    # Save the updated configuration
    return save_config()

# Initialize configuration
load_config()

# Convenience accessors for common settings
def get_model_library_path():
    return get_setting('paths.model_library')

def get_database_path():
    return get_setting('paths.database')

def get_model_extensions():
    return get_setting('models.extensions')

def get_port():
    return get_setting('app.port')

def get_debug_mode():
    return get_setting('app.debug')

def get_ollama_enabled():
    return get_setting('frameworks.ollama.enabled')

def get_ollama_repositories():
    return get_setting('frameworks.ollama.repositories', [])

def get_app_version():
    return get_setting('app.version')

def get_tool_repositories():
    return get_setting('tool_repositories', [])
