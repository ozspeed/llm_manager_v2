import os
import json
import logging
import traceback
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
            "api_token": ""
        }
    },
    "tool_repositories": []
}

_settings = None

# =============================================================================
# SETTINGS MANAGEMENT - Core functions for loading and saving settings
# =============================================================================

def load_settings() -> None:
    """Load settings from settings.json file.
    
    This function loads application settings from the settings.json file.
    If the file doesn't exist or is invalid, it creates a new one with default settings.
    It also ensures that the minimal required structure exists in the settings.
    
    Returns:
        None
    """
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
    """Save current settings to settings.json file.
    
    This function saves the current settings to the settings.json file.
    It creates any necessary parent directories if they don't exist.
    
    Returns:
        bool: True if settings were saved successfully, False otherwise
    """
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

# =============================================================================
# SETTINGS ACCESS - Functions for getting and setting values
# =============================================================================

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value by key using dot notation.
    
    Args:
        key: The key to look up, using dot notation (e.g., 'app.port')
        default: The default value to return if the key is not found
        
    Returns:
        Any: The value for the specified key, or the default if not found
    """
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
    """Update multiple settings at once.
    
    This function updates multiple settings at once by converting a flat dictionary
    with dot notation keys into a nested structure and then updating the settings.
    
    Args:
        settings_dict: A dictionary with settings to update
        
    Returns:
        bool: True if settings were updated successfully, False otherwise
    """
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

def set_setting(key: str, value: Any) -> bool:
    """Set a setting value by key using dot notation.
    
    Args:
        key: The key to set, using dot notation (e.g., 'app.port')
        value: The value to set
        
    Returns:
        bool: True if the setting was set successfully, False otherwise
    """
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
