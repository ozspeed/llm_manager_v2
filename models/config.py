import os
import json
import logging
from pathlib import Path

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
            "api_token": ""
        }
    },
    "tool_repositories": []
}

_settings = None

def load_settings():
    """Load settings from settings.json file."""
    global _settings, config
    
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
    
    # Update the config reference
    _update_config_reference()

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
        logger.info(f"Saving settings to: {settings_path}")
        
        # Log key settings before saving (for debugging)
        logger.info(f"Settings to save - Hugging Face enabled: {_settings.get('frameworks', {}).get('huggingface', {}).get('enabled')}")
        has_token = bool(_settings.get('frameworks', {}).get('huggingface', {}).get('api_token'))
        logger.info(f"Settings to save - Hugging Face API token present: {has_token}")
        
        with open(settings_path, 'w') as f:
            json.dump(_settings, f, indent=4)
        
        # Verify file was written
        if os.path.exists(settings_path):
            file_size = os.path.getsize(settings_path)
            logger.info(f"Saved configuration to settings.json (size: {file_size} bytes)")
            
            # Read back the file to verify content (for debugging)
            with open(settings_path, 'r') as f:
                saved_settings = json.load(f)
                hf_enabled = saved_settings.get('frameworks', {}).get('huggingface', {}).get('enabled')
                has_token = bool(saved_settings.get('frameworks', {}).get('huggingface', {}).get('api_token'))
                logger.info(f"Verified saved settings - Hugging Face enabled: {hf_enabled}")
                logger.info(f"Verified saved settings - Hugging Face API token present: {has_token}")
        else:
            logger.error("Failed to verify settings file after saving")
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

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
        # Log the incoming settings for debugging
        logger.info(f"Updating settings with: {json.dumps(settings_dict)}")
        
        # Check for Hugging Face settings specifically
        if 'frameworks.huggingface.enabled' in settings_dict:
            logger.info(f"Hugging Face enabled setting: {settings_dict['frameworks.huggingface.enabled']}")
        if 'frameworks.huggingface.api_token' in settings_dict:
            token_value = settings_dict['frameworks.huggingface.api_token']
            # Don't log the actual token for security, just whether it's present
            logger.info(f"Hugging Face API token provided: {bool(token_value)}")
        
        # Create a new dictionary with properly nested structure
        nested_settings = {}
        
        for key, value in settings_dict.items():
            if '.' in key:  # Handle dot notation keys
                parts = key.split('.')
                current = nested_settings
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:  # Last part
                        current[part] = value
                        logger.debug(f"Set {'.'.join(parts[:i+1])} = {value if part != 'api_token' else '[REDACTED]'}")
                    else:  # Navigate to nested dict
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            else:
                # Handle non-nested keys
                nested_settings[key] = value
                logger.debug(f"Set {key} = {value if key != 'api_token' else '[REDACTED]'}")
        
        # Now update the settings with the properly nested structure
        logger.info(f"Nested settings structure: {json.dumps(nested_settings)}")
        
        # Deep update function for nested dictionaries
        def deep_update(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value
        
        # Update the settings
        deep_update(_settings, nested_settings)
        
        # Save the updated settings
        save_settings()
        
        # Verify Hugging Face settings after update
        logger.info(f"After update - Hugging Face enabled: {get_huggingface_enabled()}")
        logger.info(f"After update - Hugging Face API token set: {bool(get_huggingface_api_token())}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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

def get_huggingface_enabled():
    return get_setting('frameworks.huggingface.enabled', DEFAULT_SETTINGS['frameworks']['huggingface']['enabled'])

def get_huggingface_api_token():
    return get_setting('frameworks.huggingface.api_token', DEFAULT_SETTINGS['frameworks']['huggingface']['api_token'])

def get_app_version():
    return get_setting('app.version', DEFAULT_SETTINGS['app']['version'])

def get_tool_repositories():
    return get_setting('tool_repositories', DEFAULT_SETTINGS['tool_repositories'])

# For compatibility with the root config.py file
def load_config():
    """Alias for load_settings for backward compatibility."""
    load_settings()

def save_config():
    """Alias for save_settings for backward compatibility."""
    save_settings()

# Make the settings accessible as config.config to maintain compatibility
# Create a config variable that points to _settings
config = _settings

# Update the config reference whenever _settings is updated
def _update_config_reference():
    global config, _settings
    config = _settings

# Load settings on module import
load_settings()
