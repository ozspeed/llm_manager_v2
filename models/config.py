import os
import json
import logging

logger = logging.getLogger(__name__)

# Default settings - only keeping model extensions as default
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
        }
    },
    "tool_repositories": []
}

_settings = None

def load_settings():
    """Load settings from settings.json file."""
    global _settings
    
    try:
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
        logger.info(f"Looking for settings at: {settings_path}")
        
        if os.path.exists(settings_path):
            logger.info("Settings file found, loading...")
            try:
                with open(settings_path, 'r') as f:
                    file_content = f.read()
                    logger.info(f"Raw settings content: {file_content[:100]}...")
                    loaded_settings = json.loads(file_content)
                    
                # Start with loaded settings and only use defaults for missing values
                _settings = loaded_settings
                logger.info(f"Loaded settings: {_settings}")
                
                # Ensure model extensions exist
                if 'models' not in _settings or 'extensions' not in _settings['models']:
                    logger.info("Adding default model extensions")
                    if 'models' not in _settings:
                        _settings['models'] = {}
                    _settings['models']['extensions'] = DEFAULT_SETTINGS['models']['extensions']
                
                # Ensure minimal structure exists
                for section in ['app', 'paths', 'frameworks']:
                    if section not in _settings:
                        logger.info(f"Adding missing section: {section}")
                        _settings[section] = {}
                        
                # Ensure frameworks.ollama exists
                if 'frameworks' in _settings and 'ollama' not in _settings['frameworks']:
                    _settings['frameworks']['ollama'] = {}
                
                logger.info("Loaded configuration from settings.json")
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON decode error: {str(json_error)}")
                logger.error(f"Invalid JSON in settings file: {settings_path}")
                _settings = DEFAULT_SETTINGS.copy()
        else:
            # If no settings file exists, use minimal defaults and create the file
            logger.info("Settings file not found, creating default")
            _settings = DEFAULT_SETTINGS.copy()
            save_settings()
            logger.info("Created default configuration")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        _settings = DEFAULT_SETTINGS.copy()  # Use defaults on error instead of empty settings

def deep_update(target, source):
    """Recursively update target dict with values from source dict."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_update(target[key], value)
        else:
            target[key] = value

def save_settings():
    """Save current settings to settings.json file."""
    try:
        settings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
        with open(settings_path, 'w') as f:
            json.dump(_settings, f, indent=4)
        logger.info("Saved configuration to settings.json")
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")

def get_setting(key, default=None):
    """Get a setting value by key using dot notation."""
    if _settings is None:
        load_settings()
    
    try:
        value = _settings
        for k in key.split('.'):
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def update_settings(settings_dict):
    """Update multiple settings at once."""
    if _settings is None:
        load_settings()
    
    try:
        def deep_update(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value
        
        deep_update(_settings, settings_dict)
        save_settings()
        return True
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return False

def set_setting(key, value):
    """Set a setting value by key using dot notation."""
    if _settings is None:
        load_settings()
    
    try:
        # Handle nested dictionaries
        if isinstance(value, dict):
            current = get_setting(key, {})
            if not isinstance(current, dict):
                current = {}
            current.update(value)
            value = current

        # Set the value
        keys = key.split('.')
        target = _settings
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            elif not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        
        # Save settings to file
        save_settings()
        return True
    except Exception as e:
        logger.error(f"Error setting configuration value: {str(e)}")
        return False

# Common getters
def get_model_library_path():
    return os.path.expanduser(get_setting('paths.model_library', DEFAULT_SETTINGS['paths']['model_library']))

def get_database_path():
    return get_setting('paths.database', DEFAULT_SETTINGS['paths']['database'])

def get_draft_download_area():
    return os.path.expanduser(get_setting('paths.draft_download_area', DEFAULT_SETTINGS['paths']['draft_download_area'] or '~/Downloads/LLM_Downloads'))

def get_model_extensions():
    return get_setting('models.extensions', DEFAULT_SETTINGS['models']['extensions'])

def get_port():
    return get_setting('app.port', DEFAULT_SETTINGS['app']['port'])

def get_debug_mode():
    return get_setting('app.debug', DEFAULT_SETTINGS['app']['debug'])

def get_ollama_enabled():
    return get_setting('frameworks.ollama.enabled', DEFAULT_SETTINGS['frameworks']['ollama']['enabled'])

def get_ollama_repositories():
    return get_setting('frameworks.ollama.repositories', DEFAULT_SETTINGS['frameworks']['ollama']['repositories'])

# Load settings on module import
load_settings()
