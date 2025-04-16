"""Main Flask application for the LLM Model Manager.

This module serves as the entry point for the application and defines all the routes.
It imports functionality from other modules to keep the code modular and maintainable.
"""

# Standard library imports
import os
import shutil
import glob
import re
import sqlite3
import json
import datetime
import traceback
import argparse
from typing import Dict, Any, Tuple, List, Optional, Union

# Third-party imports
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from functools import wraps

# Application imports - Configuration
from models import config

# Application imports - Database
from models.database import init_db, get_all_models, add_model, delete_model, reset_database

# Application imports - Model management
from models.detection import scan_directory, is_shard_file, extract_base_name
from models.ollama import scan_ollama_models
from models.huggingface import (
    search_models, 
    download_model, 
    move_model_to_library, 
    list_draft_models, 
    get_model_files,
    delete_draft_model
)

# Application imports - Utilities
from utils.system import get_system_info
from utils.file_browser import browse_directories
from utils.model_scanner import scan_for_models
from utils.error_handler import api_error_response, api_success_response, handle_api_exception

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)
# Register global error handler for all exceptions
app.register_error_handler(Exception, handle_api_exception)

# Initialize database
init_db()

# =============================================================================
# PAGE ROUTES - Main application pages
# =============================================================================

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/huggingface')
def huggingface():
    """Render the Hugging Face integration page."""
    return render_template('huggingface.html')

@app.route('/settings')
def settings():
    """Render the application settings page."""
    return render_template('settings.html')

# =============================================================================
# DEVELOPMENT & TESTING ROUTES - Only used during development
# =============================================================================

@app.route('/test')
def test():
    """Render the test page for development."""
    return render_template('index_new.html')

@app.route('/test-cancel')
def test_cancel():
    """Test page for cancel button functionality."""
    return render_template('test_cancel.html')

@app.route('/debug-cancel')
def debug_cancel():
    """Debug page for cancel button functionality."""
    return render_template('debug_cancel.html')

@app.route('/cancel-debug')
def cancel_debug():
    """Comprehensive debug tool for cancel button functionality."""
    return render_template('cancel_debug.html')

@app.route('/api-test')
def api_test():
    """Simple API test tool for testing endpoints directly."""
    return render_template('api_test.html')

# =============================================================================
# API ROUTES - MODELS - Core model management endpoints
# =============================================================================

@app.route('/api/models', methods=['GET'])
def get_models() -> Tuple[Dict[str, Any], int]:
    """Get all models from the database.
    
    Returns:
        tuple: API response with all models and status code
    """
    # Get models from database
    models, status_code = get_all_models()
    
    # Log the result
    app.logger.info(f"Retrieved {len(models) if isinstance(models, list) else 'error'} models from database")
    
    # Check if there was an error
    if status_code != 200 or isinstance(models, dict) and 'error' in models:
        error_message = models.get('error', 'Unknown error retrieving models') if isinstance(models, dict) else 'Failed to retrieve models'
        return api_error_response(error_message, status_code, "database_error")
    
    # For backward compatibility with the frontend, return the models array directly
    # instead of wrapping it in a data property
    app.logger.info(f"Returning {len(models)} models in backward-compatible format")
    return models, 200

@app.route('/api/models', methods=['POST'])
def create_model() -> Tuple[Dict[str, Any], int]:
    """Create a new model entry in the database.
    
    This endpoint adds a new model to the database with the provided information.
    All required fields must be present in the request.
    
    JSON parameters:
    - name: Model name
    - framework: Model framework
    - path: Path to the model file
    - config: Optional configuration data
    
    Returns:
        tuple: API response with the created model and status code
    """
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    name = data.get('name')
    framework = data.get('framework')
    path = data.get('path')
    config = data.get('config')
    
    # Validate required fields
    missing_fields = []
    if not name:
        missing_fields.append("name")
    if not framework:
        missing_fields.append("framework")
    if not path:
        missing_fields.append("path")
    
    if missing_fields:
        return api_error_response(
            f"Missing required fields: {', '.join(missing_fields)}", 
            400, 
            "validation_error"
        )
    
    # Log the operation
    app.logger.info(f"Creating new model: {name} ({framework}) at {path}")
    
    # Add the model to the database
    result = add_model(name, framework, path, config)
    
    # Check if the result is already a tuple with response and status code
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], int):
        return result
    
    # Otherwise, wrap it in a success response
    return api_success_response("Model created successfully", {"model": result})

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def remove_model(model_id: int) -> Tuple[Dict[str, Any], int]:
    """Remove a model from the database.
    
    This endpoint deletes a model from the database and optionally removes the associated files.
    
    Args:
        model_id: ID of the model to remove
        
    Query parameters:
    - delete_files: Whether to delete the model files (default: false)
    
    Returns:
        tuple: API response with deletion result and status code
    """
    # Check if we should delete the files as well
    delete_files = request.args.get('delete_files', 'false').lower() == 'true'
    
    # Log the operation
    app.logger.info(f"Removing model ID {model_id} (delete_files={delete_files})")
    
    # Delete the model
    result = delete_model(model_id, delete_files)
    
    # Check if the result is already a tuple with response and status code
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], int):
        return result
    
    # Otherwise, wrap it in a success response
    return api_success_response(f"Model {model_id} deleted successfully", {"result": result})

@app.route('/api/scan', methods=['GET'])
def scan_models() -> Tuple[Dict[str, Any], int]:
    """Scan for models in the specified directory and optionally include Ollama models.
    
    This endpoint scans the specified directory for model files and adds them to the database.
    It can also include Ollama models if enabled in the configuration.
    
    Query parameters:
    - path: Path to scan for models (default: model library path from config)
    - include_ollama: Whether to include Ollama models (default: based on config)
    
    Returns:
        tuple: API response with scan results and status code
    """
    # Get parameters
    path = request.args.get('path', config.get_model_library_path())
    include_ollama = request.args.get('include_ollama', str(config.get_ollama_enabled())).lower() == 'true'
    
    # Log the operation
    app.logger.info(f"Scanning for models in {path} (include_ollama={include_ollama})")
    
    # Scan filesystem models
    result, status_code = scan_directory(path)
    
    if status_code != 200:
        app.logger.error(f"Error scanning directory: {result.get('error', 'Unknown error')}")
        return api_error_response(
            result.get('error', 'Error scanning directory'), 
            status_code,
            "scan_error"
        )
    
    # Scan Ollama models if requested and enabled in config
    if include_ollama and config.get_ollama_enabled():
        app.logger.info("Including Ollama models in scan")
        ollama_result = scan_ollama_models(repositories=config.get_ollama_repositories())
        
        # Update the result with Ollama models
        if ollama_result and 'models' in ollama_result:
            # Count existing Ollama models
            existing_ollama = sum(1 for model in ollama_result['models'] if model.get('already_exists', False))
            new_ollama = len(ollama_result['models']) - existing_ollama
            
            result['added'] += new_ollama
            result['existing'] = result.get('existing', 0) + existing_ollama
            result['models'].extend(ollama_result['models'])
            result['ollama_models_found'] = len(ollama_result['models'])
            
            app.logger.info(f"Added {new_ollama} new Ollama models and found {existing_ollama} existing ones")
    
    # Return success response
    return api_success_response(
        f"Scan completed successfully. Found {result.get('added', 0)} new models and {result.get('existing', 0)} existing ones.",
        result
    )

@app.route('/api/system-info', methods=['GET'])
def system_info() -> Tuple[Dict[str, Any], int]:
    """Get system information including OS, CPU, memory, and disk usage.
    
    Returns:
        tuple: API response with system information and status code
    """
    app.logger.info("Retrieving system information")
    info, status_code = get_system_info()
    
    if status_code != 200:
        app.logger.error(f"Error getting system info: {info.get('error', 'Unknown error')}")
        return api_error_response(
            info.get('error', 'Error retrieving system information'),
            status_code,
            "system_info_error"
        )
    
    return api_success_response("System information retrieved successfully", info)

@app.route('/api/refresh-models', methods=['POST'])
def refresh_models() -> Tuple[Dict[str, Any], int]:
    """Refresh the models in the database by scanning the model library.
    
    This endpoint scans the model library directory and updates the database
    with any new or changed models. It also includes Ollama models if enabled
    in the configuration.
    
    Returns:
        tuple: API response with refresh results and status code
    """
    # Get the model library path
    model_library = config.get_model_library_path()
    app.logger.info(f"Refreshing models from {model_library}")
    
    # Scan the model library
    result, status_code = scan_directory(model_library)
    
    if status_code != 200:
        app.logger.error(f"Error scanning directory: {result.get('error', 'Unknown error')}")
        return api_error_response(
            result.get('error', 'Error refreshing models'), 
            status_code,
            "refresh_error"
        )
    
    # Include Ollama models if enabled
    if config.get_ollama_enabled():
        app.logger.info("Including Ollama models in refresh")
        ollama_result = scan_ollama_models(repositories=config.get_ollama_repositories())
        
        # Update the result with Ollama models
        if ollama_result and 'models' in ollama_result:
            # Count existing Ollama models
            existing_ollama = sum(1 for model in ollama_result['models'] if model.get('already_exists', False))
            new_ollama = len(ollama_result['models']) - existing_ollama
            
            result['added'] += new_ollama
            result['existing'] = result.get('existing', 0) + existing_ollama
            result['models'].extend(ollama_result['models'])
            result['ollama_models_found'] = len(ollama_result['models'])
            
            app.logger.info(f"Added {new_ollama} new Ollama models and found {existing_ollama} existing ones")
    
    # Return success response
    return api_success_response(
        f"Models refreshed successfully. Found {result.get('added', 0)} new models and {result.get('existing', 0)} existing ones.",
        result
    )

@app.route('/api/reset-database', methods=['POST'])
def reset_db() -> Tuple[Dict[str, Any], int]:
    """Reset the database to its initial state.
    
    This endpoint removes all model entries from the database but does not delete any files.
    
    Returns:
        tuple: API response with reset result and status code
    """
    app.logger.warning("Resetting database to initial state")
    result = reset_database()
    
    # Check if the result is already a tuple with response and status code
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], int):
        return result
    
    # Otherwise, wrap it in a success response
    return api_success_response("Database reset successfully", {"result": result})

# API endpoint to add a model with directory structure
@app.route('/api/add-model', methods=['POST'])
def add_model_with_directory() -> Tuple[Dict[str, Any], int]:
    """Add a model by copying files from source to the model library with directory structure.
    
    This endpoint copies model files from a source directory to the model library,
    preserving the directory structure. It creates the necessary directories and
    handles model files based on their extensions.
    
    JSON parameters:
    - sourceDir: Source directory containing the model files
    - modelDir: Directory containing the actual model files
    - modelName: Name to use for the model
    - framework: Optional framework identifier (default: auto-detect)
    
    Returns:
        tuple: API response with add result and status code
    """
    # Validate request data
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    source_dir = data.get('sourceDir')
    model_dir = data.get('modelDir')
    model_name = data.get('modelName')
    
    # We'll auto-detect the framework and always handle shards
    framework = data.get('framework', 'unknown')
    handle_shards = True
    
    # Validate required fields
    missing_fields = []
    if not source_dir:
        missing_fields.append("sourceDir")
    if not model_dir:
        missing_fields.append("modelDir")
    if not model_name:
        missing_fields.append("modelName")
    
    if missing_fields:
        return api_error_response(
            f"Missing required fields: {', '.join(missing_fields)}", 
            400, 
            "validation_error"
        )
    
    # Log the operation
    app.logger.info(f"Adding model '{model_name}' from {model_dir} to library")
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Check if source directory exists
    if not os.path.isdir(model_dir):
        app.logger.error(f"Model directory does not exist: {model_dir}")
        return api_error_response(f"Model directory does not exist: {model_dir}", 400, "directory_error")
    
    # Create publisher directory in model library if it doesn't exist
    publisher_name = os.path.basename(source_dir)
    publisher_dir = os.path.join(model_library, publisher_name)
    
    if not os.path.exists(publisher_dir):
        os.makedirs(publisher_dir)
        app.logger.info(f"Created publisher directory: {publisher_dir}")
    
    # Create model directory path in the library
    target_model_dir = os.path.join(publisher_dir, model_name)
    
    # Check if model directory already exists in the library
    if os.path.exists(target_model_dir):
        app.logger.warning(f"Model directory already exists: {target_model_dir}")
        return api_error_response(
            f"Model directory already exists: {target_model_dir}",
            409,
            "model_exists",
            {
                "needsConfirmation": True,
                "modelName": model_name,
                "modelPath": target_model_dir
            }
        )
    
    try:
        # Create the model directory in the library
        os.makedirs(target_model_dir)
        
        # Find model files in the source directory
        model_extensions = config.get_model_extensions()
        model_files = []
        
        for ext in model_extensions:
            pattern = os.path.join(model_dir, f"*.{ext}")
            model_files.extend(glob.glob(pattern))
        
        if not model_files:
            # Clean up the empty directory we created
            os.rmdir(target_model_dir)
            return jsonify({"error": f"No model files found in {model_dir}"}), 400
        
        # Process sharded models if enabled
        if handle_shards:
            # Group files by their base name (for sharded models)
            shard_groups = {}
            non_sharded_files = []
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if is_shard_file(filename):
                    base_name = extract_base_name(filename)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file_path)
                else:
                    non_sharded_files.append(file_path)
            
            # Copy each group of sharded files
            for base_name, shard_files in shard_groups.items():
                for shard_file in shard_files:
                    filename = os.path.basename(shard_file)
                    target_path = os.path.join(target_model_dir, filename)
                    print(f"Copying shard file: {shard_file} -> {target_path}")
                    shutil.copy2(shard_file, target_path)
            
            # Copy non-sharded files
            for file_path in non_sharded_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
        else:
            # Copy all files without shard processing
            for file_path in model_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
        
        # Add the model to the database
        # For sharded models, use the virtual path to the directory
        if handle_shards and shard_groups:
            # Add each sharded model group to the database
            for base_name, shard_files in shard_groups.items():
                # Calculate total size of all shards
                total_size_mb = sum(os.path.getsize(f) for f in shard_files) / (1024 * 1024)
                
                # Create a friendly name from the base name
                friendly_name = f"{model_name} - {base_name}"
                
                # Create a virtual path to the model directory
                virtual_path = os.path.join(target_model_dir, base_name)
                
                # Add to database with shard information
                result, status_code = add_model(
                    name=friendly_name,
                    framework=framework,
                    path=virtual_path,
                    model_config={
                        "is_sharded": True,
                        "shard_count": len(shard_files),
                        "publisher": publisher_name,
                        "model_dir": model_name
                    },
                    size_override=total_size_mb
                )
                
                if status_code != 201:
                    return jsonify({
                        "error": f"Failed to add sharded model to database: {result.get('error', 'Unknown error')}"
                    }), status_code
        
        # Add non-sharded models to the database
        for file_path in non_sharded_files if handle_shards else model_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_model_dir, filename)
            
            # Calculate file size
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            
            # Add to database
            result, status_code = add_model(
                name=f"{model_name} - {filename}",
                framework=framework,
                path=target_path,
                model_config={
                    "is_sharded": False,
                    "publisher": publisher_name,
                    "model_dir": model_name
                },
                size_override=size_mb
            )
            
            if status_code != 201:
                return jsonify({
                    "error": f"Failed to add model to database: {result.get('error', 'Unknown error')}"
                }), status_code
        
        return jsonify({
            "message": f"Successfully added model(s) from {model_name}",
            "modelPath": target_model_dir
        }), 201
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(target_model_dir):
            shutil.rmtree(target_model_dir)
        return jsonify({"error": str(e)}), 500

# API endpoint to confirm model overwrite
@app.route('/api/confirm-overwrite', methods=['POST'])
def confirm_overwrite() -> Tuple[Dict[str, Any], int]:
    """Confirm overwriting an existing model.
    
    This endpoint handles the confirmation process for overwriting an existing model.
    It deletes the existing model from the database and filesystem, then adds the new model.
    
    JSON parameters:
    - sourceDir: Source directory containing the model files
    - modelDir: Directory containing the actual model files
    - modelName: Name of the model
    - framework: Framework identifier
    - handleShards: Whether to handle sharded models (default: true)
    - confirm: Whether to proceed with the overwrite (default: false)
    
    Returns:
        tuple: API response with overwrite result and status code
    """
    # Validate request data
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    source_dir = data.get('sourceDir')
    model_dir = data.get('modelDir')
    model_name = data.get('modelName')
    framework = data.get('framework')
    handle_shards = data.get('handleShards', True)
    confirm = data.get('confirm', False)
    
    # Check if operation was cancelled
    if not confirm:
        app.logger.info(f"Overwrite operation cancelled for model: {model_name}")
        return api_success_response("Operation cancelled")
    
    # Validate required fields
    missing_fields = []
    if not source_dir:
        missing_fields.append("sourceDir")
    if not model_dir:
        missing_fields.append("modelDir")
    if not model_name:
        missing_fields.append("modelName")
    if not framework:
        missing_fields.append("framework")
    
    if missing_fields:
        return api_error_response(
            f"Missing required fields: {', '.join(missing_fields)}", 
            400, 
            "validation_error"
        )
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Determine the target model directory
    publisher_name = os.path.basename(source_dir)
    publisher_dir = os.path.join(model_library, publisher_name)
    target_model_dir = os.path.join(publisher_dir, model_name)
    
    # Log the operation
    app.logger.info(f"Confirming overwrite for model: {model_name} in {target_model_dir}")
    
    # Find all models in the database with this path prefix
    conn = sqlite3.connect(config.get_database_path())
    cursor = conn.cursor()
    
    # Use LIKE with path prefix to find all related models
    cursor.execute("SELECT id FROM models WHERE path LIKE ?", (f"{target_model_dir}%",))
    model_ids = [row[0] for row in cursor.fetchall()]
    app.logger.info(f"Found {len(model_ids)} model entries to delete")
    
    # Delete models from database
    for model_id in model_ids:
        app.logger.info(f"Deleting model ID {model_id} from database")
        delete_model(model_id)
    
    # Delete the directory and its contents
    if os.path.exists(target_model_dir):
        app.logger.info(f"Deleting directory: {target_model_dir}")
        shutil.rmtree(target_model_dir)
    
    # Now proceed with adding the model again
    app.logger.info(f"Proceeding to add model: {model_name}")
    return add_model_with_directory()

# API endpoint to get configuration
@app.route('/api/config', methods=['GET'])
def get_config() -> Tuple[Dict[str, Any], int]:
    """Get the current configuration.
    
    This endpoint returns the current application configuration settings.
    It ensures that all required configuration sections exist and have valid values.
    
    Returns:
        tuple: API response with configuration settings and status code
    """
    app.logger.info("Config API endpoint called")
    
    # Load settings if not already loaded
    if config._settings is None:
        app.logger.info("Settings not loaded, loading now")
        config.load_settings()
    
    # Ensure we have a valid settings object
    if not config._settings:
        app.logger.warning("Empty settings detected, reloading defaults")
        config._settings = config.DEFAULT_SETTINGS.copy()
        
    # Log the settings for debugging
    app.logger.debug(f"Returning settings: {config._settings}")
    
    # Ensure the structure is complete
    if 'paths' not in config._settings:
        config._settings['paths'] = {}
    if 'models' not in config._settings:
        config._settings['models'] = {'extensions': config.DEFAULT_SETTINGS['models']['extensions']}
    if 'app' not in config._settings:
        config._settings['app'] = {}
    if 'frameworks' not in config._settings:
        config._settings['frameworks'] = {'ollama': {}}
    
    # Return success response with settings
    return api_success_response("Configuration retrieved successfully", config._settings)

# API endpoint to browse directories
@app.route('/api/browse', methods=['GET'])
def browse_dir() -> Tuple[Dict[str, Any], int]:
    """Browse directories for the directory selector.
    
    This endpoint provides a directory listing for the file browser component.
    It returns a list of subdirectories and files in the specified path.
    
    Query parameters:
    - path: Directory path to browse (default: '/')
    
    Returns:
        tuple: API response with directory contents and status code
    """
    path = request.args.get('path', '/')
    app.logger.info(f"Browsing directory: {path}")
    
    result = browse_directories(path)
    
    # Check if the result is already a tuple with response and status code
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and isinstance(result[1], int):
        return result
    
    # Return the raw data as-is to match frontend expectations
    # The data should already be structured as expected: { directories: [...], current_path: '...', parent_dir: '...' }
    return result

# API endpoint to update configuration
@app.route('/api/config', methods=['POST'])
def update_config() -> Tuple[Dict[str, Any], int]:
    """Update configuration settings.
    
    This endpoint updates the application configuration with the provided settings.
    It saves the updated configuration to the settings file.
    
    JSON parameters:
    - Any valid configuration settings to update
    
    Returns:
        tuple: API response with updated configuration and status code
    """
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    app.logger.info(f"Updating configuration with: {data}")
    
    # Update all settings at once
    if config.update_settings(data):
        app.logger.info("Configuration updated successfully")
        return api_success_response(
            "Configuration updated successfully", 
            {"config": config._settings}
        )
    else:
        app.logger.error("Failed to update configuration")
        return api_error_response(
            "Failed to update configuration", 
            500, 
            "config_update_error"
        )

# API endpoint to sync repository with AI Library
@app.route('/api/sync_repository', methods=['POST'])
def sync_repository() -> Tuple[Dict[str, Any], int]:
    """Sync repository with AI Library by creating symlinks based on repository settings.
    
    This endpoint creates symlinks from the model library to a repository location,
    allowing models to be used by other applications. It handles different repository
    requirements and model types.
    
    JSON parameters:
    - name: Repository name
    - location: Repository location (directory path)
    - require_parent_directory: Whether the repository requires a parent directory structure
    - understands_shards: Whether the repository understands sharded models
    
    Returns:
        tuple: API response with sync results and status code
    """
    # Validate request data
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    name = data.get('name')
    location = data.get('location')
    require_parent_directory = data.get('require_parent_directory', False)
    understands_shards = data.get('understands_shards', False)
    
    # Validate required fields
    missing_fields = []
    if not name:
        missing_fields.append("name")
    if not location:
        missing_fields.append("location")
    
    if missing_fields:
        return api_error_response(
            f"Missing required fields: {', '.join(missing_fields)}", 
            400, 
            "validation_error"
        )
    
    # Log the operation
    app.logger.info(f"Syncing repository '{name}' at {location} (require_parent_directory={require_parent_directory}, understands_shards={understands_shards})")
    
    # Get the model library path
    model_library_path = config.get_model_library_path()
    
    if not model_library_path:
        app.logger.error("Model library path is not set")
        return api_error_response("Model library path is not set", 400, "config_error")
    
    # Import necessary modules
    import os
    import shutil
    import datetime
    import re
    from pathlib import Path
    
    # Create symlinks from model library to repository location
    model_count = 0
    skipped_count = 0
    sharded_count = 0
    model_extensions = config.get_model_extensions()
    
    # Ensure repository directory exists
    os.makedirs(location, exist_ok=True)
    
    # Track models by type
    model_types = {}
    
    # First, prescan the target directory to find existing symlinks
    existing_symlinks = {}
    for root, dirs, files in os.walk(location):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.islink(file_path):
                # Get the real path that the symlink points to
                try:
                    real_path = os.path.realpath(file_path)
                    # Store the real path as a key to avoid duplicate symlinks
                    existing_symlinks[real_path] = file_path
                except Exception:
                    # If we can't resolve the symlink, just continue
                    pass
    
    # Scan model library for models
    for ext in model_extensions:
        for model_file in Path(model_library_path).glob(f'**/*{ext}'):
            # Skip sharded models if the repository doesn't understand them
            # More robust shard detection - looking for patterns like .00.gguf, .01.gguf, etc.
            is_shard = False
            file_name = model_file.name
            
            # Check for common shard patterns
            # Pattern 1: name.00.ext, name.01.ext, etc.
            if re.search(r'\.[0-9]{2,}\.[^.]+$', file_name):
                is_shard = True
            # Pattern 2: name-00-of-03.ext, name-01-of-03.ext, etc.
            elif re.search(r'-[0-9]{2,}-of-[0-9]{2,}\.[^.]+$', file_name):
                is_shard = True
            # Pattern 3: name.shard0.ext, name.shard1.ext, etc.
            elif re.search(r'\.shard[0-9]+\.[^.]+$', file_name):
                is_shard = True
            
            if is_shard and not understands_shards:
                sharded_count += 1
                continue
                
            # Check if this model is already symlinked somewhere in the target directory
            real_model_path = str(model_file.resolve())
            if real_model_path in existing_symlinks:
                skipped_count += 1
                continue
            
            # Determine the target path based on whether parent directories are required
            if require_parent_directory:
                # Get the relative path from the model library to the model file's parent
                rel_path = os.path.relpath(model_file.parent, model_library_path)
                if rel_path == ".":
                    # If the model is directly in the model library, just use the repository location
                    target_dir = location
                else:
                    # Create the same directory structure in the repository location
                    target_dir = os.path.join(location, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
            else:
                # Put all models directly in the repository location
                target_dir = location
            
            target_path = os.path.join(target_dir, model_file.name)
            
            # Check if target already exists
            if os.path.exists(target_path):
                if os.path.islink(target_path):
                    # Skip existing symlinks but count them
                    skipped_count += 1
                    continue
                else:
                    # Skip real files
                    continue
            
            # Create the symlink with better error handling
            try:
                # Double-check if the target exists right before creating the symlink
                if os.path.exists(target_path):
                    # If it's a symlink, count it as skipped
                    if os.path.islink(target_path):
                        skipped_count += 1
                    # If it's a real file, we'll also skip it
                    continue
                
                # Create the symlink
                os.symlink(model_file, target_path)
                model_count += 1
                
                # Track model type for audit
                model_type = ext.lstrip('.')
                if model_type in model_types:
                    model_types[model_type] += 1
                else:
                    model_types[model_type] = 1
            except FileExistsError:
                # If the file exists despite our checks (race condition), count it as skipped
                skipped_count += 1
            except Exception as e:
                # Log other errors but continue processing
                app.logger.error(f"Error creating symlink for {model_file}: {str(e)}")
                continue
    
    # Create audit trail
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_types_str = ", ".join([f"{count} {type}" for type, count in model_types.items()])
    audit_message = f"[{timestamp}] Synchronized {model_count} models ({model_types_str})"
    
    if skipped_count > 0:
        audit_message += f", skipped {skipped_count} existing symlinks"
    
    if sharded_count > 0 and not understands_shards:
        audit_message += f", ignored {sharded_count} sharded models"
    
    # Get the current tool repositories from config
    tool_repositories = config.config.get("tool_repositories", [])
    
    # Find and update the repository with the audit trail
    for repo in tool_repositories:
        if repo.get("name") == name:
            # Overwrite the audit trail with just the current message
            repo["audit_trail"] = [audit_message]
            break
    
    # Save the updated config
    config.config["tool_repositories"] = tool_repositories
    config.save_config()
    
    app.logger.info(f"Repository sync completed: {audit_message}")
    
    # Return success response
    return api_success_response(
        f"Successfully synchronized {model_count} models for {name}",
        {
            "count": model_count,
            "skipped": skipped_count,
            "sharded": sharded_count,
            "audit": audit_message
        }
    )


# API endpoint to scan for models in a directory
@app.route('/api/scan-models-in-dir', methods=['POST'])
def scan_models_in_directory() -> Tuple[Dict[str, Any], int]:
    """Scan a directory for models and return a list of discovered models.
    
    This endpoint scans a specified directory for model files and returns a list of
    discovered models with their details. It auto-detects the framework and handles
    sharded models.
    
    JSON parameters:
    - searchStartpoint: Directory path to scan for models
    
    Returns:
        tuple: API response with discovered models and status code
    """
    # Validate request data
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    search_startpoint = data.get('searchStartpoint')
    
    # Validate required fields
    if not search_startpoint:
        return api_error_response("Missing required field: searchStartpoint", 400, "validation_error")
    
    # Log the operation
    app.logger.info(f"Scanning for models in directory: {search_startpoint}")
    
    # Check if directory exists
    if not os.path.isdir(search_startpoint):
        app.logger.error(f"Directory does not exist: {search_startpoint}")
        return api_error_response(f"Directory does not exist: {search_startpoint}", 400, "directory_error")
    
    # Scan for models
    models, errors = scan_for_models(search_startpoint)
    app.logger.info(f"Found {len(models)} models in {search_startpoint}")
    
    # Return results with any warnings
    if errors:
        app.logger.warning(f"Scan completed with {len(errors)} warnings")
        return api_success_response(
            f"Scan completed with warnings. Found {len(models)} models.",
            {
                "models": models,
                "warnings": errors
            }
        )
    
    return api_success_response(f"Scan completed successfully. Found {len(models)} models.", {"models": models})

# API endpoint to add multiple models
@app.route('/api/add-models', methods=['POST'])
def add_multiple_models():
    """Add multiple models to the library."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    models = data.get('models', [])
    
    # Log the received models data
    print("Received models data:", json.dumps(models, indent=2))
    
    if not models:
        return jsonify({"error": "Missing required field: models"}), 400
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Check for existing models
    models_to_overwrite = []
    for model in models:
        model_path = model.get('path')
        model_name = model.get('name')
        
        if not model_path or not model_name:
            continue
        
        # Use provided publisher_name or extract from path
        publisher_name = model.get('publisher_name')
        if not publisher_name:
            publisher_name = os.path.basename(os.path.dirname(model_path))
        target_model_dir = os.path.join(model_library, publisher_name, model_name)
        
        if os.path.exists(target_model_dir):
            models_to_overwrite.append(f"{publisher_name}/{model_name}")
    
    if models_to_overwrite:
        return jsonify({
            "error": "Some models already exist in the library",
            "needsConfirmation": True,
            "modelsToOverwrite": models_to_overwrite
        }), 409
    
    # Process each model
    added_models = []
    errors = []
    framework = data.get('framework', 'unknown')
    
    for model in models:
        model_path = model.get('path')
        model_name = model.get('name')
        
        if not model_path or not model_name:
            errors.append(f"Missing path or name for model: {model}")
            continue
        
        # Use provided publisher_name or extract from path
        publisher_name = model.get("publisher_name")
        print(f"Publisher name from model data: {publisher_name}")
        if publisher_name is None or publisher_name == "":
            publisher_name = os.path.basename(os.path.dirname(model_path))
            print(f"Using extracted publisher name: {publisher_name}")
        print(f"Final publisher name: {publisher_name}")
        publisher_dir = os.path.join(model_library, publisher_name)
        
        # Create publisher directory if it doesn't exist
        if not os.path.exists(publisher_dir):
            os.makedirs(publisher_dir)
        
        # Create model directory path in the library
        target_model_dir = os.path.join(publisher_dir, model_name)
        
        try:
            # Create the model directory in the library
            print(f"Creating model directory: {target_model_dir}")
            os.makedirs(target_model_dir)
            
            # Find model files in the source directory
            model_extensions = config.get_model_extensions()
            model_files = []
            
            for ext in model_extensions:
                # Remove the leading dot if present to avoid double dots in the pattern
                ext_clean = ext[1:] if ext.startswith('.') else ext
                pattern = os.path.join(model_path, f"*.{ext_clean}")
                model_files.extend(glob.glob(pattern))
            
            if not model_files:
                # Clean up the empty directory we created
                os.rmdir(target_model_dir)
                errors.append(f"No model files found in {model_path}")
                continue
            
            # Always process sharded models
            # Group files by their base name (for sharded models)
            shard_groups = {}
            non_sharded_files = []
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if is_shard_file(filename):
                    base_name = extract_base_name(filename)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file_path)
                else:
                    non_sharded_files.append(file_path)
            
            # Copy each group of sharded files
            for base_name, shard_files in shard_groups.items():
                for shard_file in shard_files:
                    filename = os.path.basename(shard_file)
                    target_path = os.path.join(target_model_dir, filename)
                    print(f"Copying shard file: {shard_file} -> {target_path}")
                    shutil.copy2(shard_file, target_path)
            
            # Copy non-sharded files
            for file_path in non_sharded_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
            
            # Add the model to the database
            if shard_groups:
                # For sharded models, add each group
                for base_name, _ in shard_groups.items():
                    model_result, _ = add_model(
                        f"{model_name} ({base_name})", 
                        framework, 
                        target_model_dir,
                        {"is_sharded": True, "base_name": base_name}
                    )
                    added_models.append(model_result)
            else:
                # For non-sharded models
                # Add the model to the database
                model_result, _ = add_model(model_name, framework, target_model_dir)
                added_models.append(model_result)
                
        except Exception as e:
            errors.append(f"Error processing model {model_name}: {str(e)}")
            # Try to clean up if an error occurred
            if os.path.exists(target_model_dir):
                try:
                    shutil.rmtree(target_model_dir)
                except:
                    pass
    
    return jsonify({
        "message": f"Added {len(added_models)} models successfully",
        "added": added_models,
        "errors": errors
    })

# API endpoint to confirm overwrite of multiple models
@app.route('/api/confirm-overwrite-multiple', methods=['POST'])
def confirm_overwrite_multiple():
    """Confirm overwriting multiple existing models."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    models = data.get('models', [])
    framework = data.get('framework')
    handle_shards = data.get('handleShards', True)
    confirm = data.get('confirm', False)
    
    if not models or not framework:
        return jsonify({"error": "Missing required fields: models, framework"}), 400
    
    if not confirm:
        return jsonify({"error": "Confirmation is required"}), 400
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Process each model
    added_models = []
    errors = []
    framework = data.get('framework', 'unknown')
    
    for model in models:
        model_path = model.get('path')
        model_name = model.get('name')
        
        if not model_path or not model_name:
            errors.append(f"Missing path or name for model: {model}")
            continue
        
        # Use provided publisher_name or extract from path
        publisher_name = model.get("publisher_name")
        print(f"Publisher name from model data: {publisher_name}")
        if publisher_name is None or publisher_name == "":
            publisher_name = os.path.basename(os.path.dirname(model_path))
            print(f"Using extracted publisher name: {publisher_name}")
        print(f"Final publisher name: {publisher_name}")
        publisher_dir = os.path.join(model_library, publisher_name)
        
        # Create publisher directory if it doesn't exist
        if not os.path.exists(publisher_dir):
            os.makedirs(publisher_dir)
        
        # Create model directory path in the library
        target_model_dir = os.path.join(publisher_dir, model_name)
        
        try:
            # Remove existing model directory if it exists
            if os.path.exists(target_model_dir):
                # First, remove from database
                conn = sqlite3.connect(config.get_database_path())
                cursor = conn.cursor()
                cursor.execute("DELETE FROM models WHERE path LIKE ?", (f"{target_model_dir}%",))
                conn.commit()
                conn.close()
                
                # Then remove the directory
                shutil.rmtree(target_model_dir)
            
            # Create the model directory in the library
            print(f"Creating model directory: {target_model_dir}")
            os.makedirs(target_model_dir)
            
            # Find model files in the source directory
            model_extensions = config.get_model_extensions()
            model_files = []
            
            for ext in model_extensions:
                # Remove the leading dot if present to avoid double dots in the pattern
                ext_clean = ext[1:] if ext.startswith('.') else ext
                pattern = os.path.join(model_path, f"*.{ext_clean}")
                model_files.extend(glob.glob(pattern))
            
            if not model_files:
                # Clean up the empty directory we created
                os.rmdir(target_model_dir)
                errors.append(f"No model files found in {model_path}")
                continue
            
            # Always process sharded models
            # Group files by their base name (for sharded models)
            shard_groups = {}
            non_sharded_files = []
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if is_shard_file(filename):
                    base_name = extract_base_name(filename)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file_path)
                else:
                    non_sharded_files.append(file_path)
            
            # Copy each group of sharded files
            for base_name, shard_files in shard_groups.items():
                for shard_file in shard_files:
                    filename = os.path.basename(shard_file)
                    target_path = os.path.join(target_model_dir, filename)
                    print(f"Copying shard file: {shard_file} -> {target_path}")
                    shutil.copy2(shard_file, target_path)
            
            # Copy non-sharded files
            for file_path in non_sharded_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
            
            # Add the model to the database
            if shard_groups:
                # For sharded models, add each group
                for base_name, _ in shard_groups.items():
                    model_result, _ = add_model(
                        f"{model_name} ({base_name})", 
                        framework, 
                        target_model_dir,
                        {"is_sharded": True, "base_name": base_name}
                    )
                    added_models.append(model_result)
            else:
                # For non-sharded models
                # Add the model to the database
                model_result, _ = add_model(model_name, framework, target_model_dir)
                added_models.append(model_result)
                
        except Exception as e:
            errors.append(f"Error processing model {model_name}: {str(e)}")
            # Try to clean up if an error occurred
            if os.path.exists(target_model_dir):
                try:
                    shutil.rmtree(target_model_dir)
                except:
                    pass
    
    return jsonify({
        "message": f"Overwritten and added {len(added_models)} models successfully",
        "added": added_models,
        "errors": errors
    })





# API endpoint for server status
@app.route('/api/status', methods=['GET'])
def server_status():
    """Enhanced endpoint to check if the server is running with detailed status information."""
    import platform
    import psutil
    import os
    
    # Get process info
    process = psutil.Process(os.getpid())
    
    try:
        return jsonify({
            "status": "ok",
            "version": config.get_app_version(),
            "timestamp": datetime.datetime.now().isoformat(),
            "uptime": round((datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())).total_seconds()),
            "port": config.get_port(),
            "debug": config.get_debug_mode(),
            "platform": platform.system(),
            "python_version": platform.python_version()
        })
    except Exception as e:
        # Fallback to simple response if detailed info fails
        return jsonify({
            "status": "ok",
            "version": config.get_app_version(),
            "timestamp": datetime.datetime.now().isoformat()
        })


@app.route('/api/server/stop', methods=['POST'])
def stop_server():
    """Stop the server gracefully.
    
    This endpoint triggers a graceful shutdown of the current server.
    """
    import os
    import threading
    import time
    
    # Function to run the stop process after response is sent
    def run_stop():
        try:
            # Give time for the response to be sent
            time.sleep(1)
            # Log success and exit
            print("Shutting down server process")
            os._exit(0)  # Force immediate exit
        except Exception as e:
            print(f"Error during server shutdown: {str(e)}")
    
    try:
        # Start the stop process in a separate thread
        # This ensures the response is sent before the server exits
        stop_thread = threading.Thread(target=run_stop)
        stop_thread.daemon = True
        stop_thread.start()
        
        # Return success response
        return jsonify({
            "success": True, 
            "message": "Server shutdown initiated", 
            "details": "The server will shut down momentarily."
        })
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e), 
            "message": "Failed to stop server."
        })

# Hugging Face API endpoints
@app.route('/api/huggingface/search', methods=['GET'])
def hf_search() -> Tuple[Dict[str, Any], int]:
    """Search for models on Hugging Face.
    
    Query parameters:
    - query: Search query
    - model_type: Type of model to search for (optional)
    - limit: Maximum number of results to return (optional, default: 50)
    - tags: Comma-separated list of tags to filter by (optional, default: gguf)
    - sort_by: Field to sort results by (optional, default: downloads)
    - sort_direction: Sort direction (-1 for descending, 1 for ascending, default: -1)
    - is_trending: Whether this is a trending models search (optional, default: false)
    
    Returns:
        tuple: API response with search results and status code
    """
    query = request.args.get('query', '')
    model_type = request.args.get('model_type', None)
    limit = int(request.args.get('limit', 50))
    
    # Parse optional parameters with defaults
    tags_param = request.args.get('tags', 'gguf')
    tags = tags_param.split(',') if tags_param else None
    
    sort_by = request.args.get('sort_by', 'downloads')
    try:
        sort_direction = int(request.args.get('sort_direction', '-1'))
    except ValueError:
        sort_direction = -1  # Default to descending if invalid value
    
    is_trending_str = request.args.get('is_trending', 'false').lower()
    is_trending = is_trending_str in ('true', 'yes', '1')
    
    app.logger.info(f"Search request: query='{query}', model_type={model_type}, sort_by={sort_by}, is_trending={is_trending}")
    
    if not query and not is_trending:
        return api_error_response("Search query is required unless is_trending=true", 400)
    
    # The search_models function now handles saving the search to history internally
    # when is_trending=True it will save as "trending", otherwise as the query
    
    from models.huggingface import search_models
    
    # Determine if we should allow fallback to trending models
    # Only allow fallback if we don't have trending models in cache
    no_fallback = False
    
    if not is_trending:
        from models.search_history import get_popular_models
        cached_trending_models = get_popular_models()
        if cached_trending_models:
            # We have trending models in cache, so don't fall back to trending on search failure
            no_fallback = True
    
    results = search_models(
        query=query, 
        model_type=model_type, 
        limit=limit,
        tags=tags,
        sort_by=sort_by,
        sort_direction=sort_direction,
        is_trending=is_trending,
        no_fallback=no_fallback
    )
    
    app.logger.info(f"Search returned {len(results)} results")
    
    return api_success_response(
        message="Search completed successfully",
        data={
            "results": results,
            "count": len(results)
        }
    )

@app.route('/api/huggingface/popular-models', methods=['GET'])
def hf_popular_models() -> Tuple[Dict[str, Any], int]:
    """Get trending models from Hugging Face.
    
    Query parameters:
    - model_type: Type of model to filter by (optional)
    - limit: Maximum number of results to return (optional, default: 12)
    
    Returns:
        tuple: API response with trending models and status code
    """
    model_type = request.args.get('model_type', None)
    
    # Get the configured search results limit from settings
    from models.config import get_setting
    configured_limit = get_setting("frameworks.huggingface.search_results_limit", default=50)
    
    # Use the query parameter limit if provided, otherwise use the configured limit
    limit = int(request.args.get('limit', configured_limit))
    
    app.logger.info(f"Retrieving trending models (limit: {limit}, model_type: {model_type})")
    
    from models.huggingface import get_trending_models
    results = get_trending_models(limit, model_type)
    
    app.logger.info(f"Retrieved {len(results)} trending models")
    
    return api_success_response(
        message="Trending models retrieved successfully",
        data={
            "results": results,
            "count": len(results)
        }
    )

@app.route('/api/huggingface/recent-searches', methods=['GET'])
def hf_recent_searches() -> Tuple[Dict[str, Any], int]:
    """Get recent search queries.
    
    Query parameters:
    - limit: Maximum number of searches to return (optional, default: 5)
    
    Returns:
        tuple: API response with recent searches and status code
    """
    limit = int(request.args.get('limit', 5))
    
    from models.search_history import get_recent_searches
    results = get_recent_searches(limit)
    
    return api_success_response(
        message="Recent searches retrieved successfully",
        data={
            "searches": results,
            "count": len(results)
        }
    )

@app.route('/api/huggingface/download', methods=['POST'])
def hf_download() -> Tuple[Dict[str, Any], int]:
    """Download a model from Hugging Face to the draft download area.
    
    JSON parameters:
    - model_id: Hugging Face model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
    - filename: Specific filename to download (optional)
    
    Returns:
        tuple: API response with download result and status code
    """
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
        
    model_id = data.get('model_id')
    filename = data.get('filename')
    
    if not model_id:
        return api_error_response("Model ID is required", 400)
    
    result = download_model(model_id, filename)
    if not result["success"]:
        error_msg = result.get('error', 'Unknown error')
        return api_error_response(error_msg, 400, "download_error", result)
    
    # Return the original result structure to maintain compatibility
    return result, 200

@app.route('/api/huggingface/draft-models', methods=['GET'])
def hf_list_draft_models() -> Tuple[Dict[str, Any], int]:
    """List all models in the draft download area.
    
    Returns:
        tuple: API response with list of draft models and status code
    """
    try:
        # Get the list of draft models
        file_models = list_draft_models()
        
        # Log the result
        app.logger.info(f"Found {len(file_models)} model files in draft area")
        
        # Transform the file-level models into model-level entries
        # Group files by publisher/model combination
        model_groups = {}
        for file_model in file_models:
            key = f"{file_model['publisher']}/{file_model['model']}"
            if key not in model_groups:
                model_dir_path = os.path.dirname(file_model['path'])
                model_groups[key] = {
                    "id": key,  # Use the publisher/model as the ID for frontend compatibility
                    "name": f"{file_model['publisher']}/{file_model['model']}",
                    "modelDirPath": model_dir_path,
                    "path": model_dir_path,  # Add path field for frontend compatibility
                    "files": [],
                    "size": 0
                }
            
            # Add file to the model's files list
            model_groups[key]["files"].append({
                "name": file_model["filename"],
                "size": file_model["size"],
                "path": file_model["path"]
            })
            
            # Update total model size
            model_groups[key]["size"] += file_model["size"]
        
        # Convert the grouped models to a list
        grouped_models = list(model_groups.values())
        
        app.logger.info(f"Grouped into {len(grouped_models)} models for frontend display")
        
        # For backward compatibility with the frontend
        # The frontend expects {success: true, models: [...]} format
        return {
            "success": True,
            "models": grouped_models
        }, 200
    except Exception as e:
        app.logger.error(f"Error listing draft models: {str(e)}")
        return {"success": False, "error": str(e)}, 500

@app.route('/api/huggingface/files', methods=['GET'])
def hf_get_files() -> Tuple[Dict[str, Any], int]:
    """Get files for a specific model on Hugging Face.
    
    Query parameters:
    - model_id: Hugging Face model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
    
    Returns:
        tuple: API response with model files and status code
    """
    model_id = request.args.get('model_id')
    if not model_id:
        return api_error_response("No model_id provided", 400)
    
    result = get_model_files(model_id)
    
    if not result.get('success', False):
        error_msg = result.get('error', 'Unknown error')
        return api_error_response(error_msg, 400, "model_files_error")
    
    # Return the original result structure to maintain compatibility
    return result, 200

@app.route('/api/huggingface/move-to-library', methods=['POST'])
def hf_move_to_library() -> Tuple[Dict[str, Any], int]:
    """Move a model from the draft download area to the model library.
    
    This endpoint moves a model from the draft download area to the model library,
    preserving the directory structure (publisher/model). It handles both file paths
    and directory paths.
    
    JSON parameters:
    - filepath: Full path to the file or directory to move
    
    Returns:
        tuple: API response with move result and status code
    """
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    filepath = data.get('filepath')
    if not filepath:
        return api_error_response("File path is required", 400)
    
    # Log the operation
    app.logger.info(f"Moving model from draft area to library: {filepath}")
    
    # Check if the path is a directory or a file
    if os.path.isdir(filepath):
        # If it's a directory, find all model files in the directory
        app.logger.info(f"Path is a directory, finding model files in: {filepath}")
        model_extensions = config.get_model_extensions()
        model_files = []
        
        for file in os.listdir(filepath):
            file_path = os.path.join(filepath, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in model_extensions):
                model_files.append(file_path)
        
        if not model_files:
            return api_error_response(f"No model files found in directory: {filepath}", 400, "no_models_found")
        
        # Move the first model file (we'll use this as the primary model)
        app.logger.info(f"Moving primary model file: {model_files[0]}")
        result = move_model_to_library(model_files[0])
        
        # Move any additional model files
        for i in range(1, len(model_files)):
            try:
                app.logger.info(f"Moving additional model file: {model_files[i]}")
                move_model_to_library(model_files[i])
            except Exception as e:
                app.logger.warning(f"Error moving additional file {model_files[i]}: {str(e)}")
    else:
        # If it's a file, move it directly
        result = move_model_to_library(filepath)
    
    if not result.get("success", False):
        error_msg = result.get('error', 'Unknown error')
        app.logger.error(f"Error moving model to library: {error_msg}")
        return api_error_response(error_msg, 400, "move_error", result)
    
    app.logger.info(f"Successfully moved model to library: {result.get('message', 'No message')}")
    
    # Return the original result structure to maintain compatibility
    return result, 200

@app.route('/api/huggingface/delete-draft', methods=['POST'])
def hf_delete_draft() -> Tuple[Dict[str, Any], int]:
    """Delete a model from the draft download area without adding it to the library.
    
    This endpoint deletes a model file or directory from the draft download area.
    It performs several safety checks to ensure the path exists and is within the
    draft area before deletion.
    
    JSON parameters:
    - filepath: Full path to the file or directory to delete
    
    Returns:
        tuple: API response with delete result and status code
    """
    app.logger.info("[HF_DELETE_DRAFT] API endpoint called")
    
    # Check if request has JSON content
    if not request.is_json:
        app.logger.error("[HF_DELETE_DRAFT] Request does not contain JSON data")
        return api_error_response("Request must be JSON", 400)
    
    # Get the JSON data
    data = request.json
    app.logger.info(f"[HF_DELETE_DRAFT] Parsed JSON data: {data}")
    
    # Extract and validate filepath
    filepath = data.get('filepath')
    app.logger.info(f"[HF_DELETE_DRAFT] Extracted filepath: {filepath}")
    
    if not filepath:
        app.logger.error("[HF_DELETE_DRAFT] No filepath provided in delete request")
        return api_error_response("File path is required", 400)
    
    if not isinstance(filepath, str):
        app.logger.error(f"[HF_DELETE_DRAFT] Invalid filepath type: {type(filepath)}")
        return api_error_response(f"Invalid filepath type: {type(filepath)}", 400)
    
    # Check if path exists
    if not os.path.exists(filepath):
        app.logger.error(f"[HF_DELETE_DRAFT] Path not found: {filepath}")
        return api_error_response(f"Path not found: {filepath}", 404)
    
    # Check if path is within draft area
    from models.config import get_draft_download_area
    draft_area = get_draft_download_area()
    app.logger.info(f"[HF_DELETE_DRAFT] Draft area: {draft_area}")
    
    if not draft_area:
        app.logger.error("[HF_DELETE_DRAFT] Draft download area not configured")
        return api_error_response("Draft download area not configured", 500)
    
    if not filepath.startswith(draft_area):
        app.logger.error(f"[HF_DELETE_DRAFT] Security check failed - Path is not within draft area")
        app.logger.error(f"[HF_DELETE_DRAFT] Path: {filepath}")
        app.logger.error(f"[HF_DELETE_DRAFT] Draft area: {draft_area}")
        return api_error_response("Security error: Path is not within draft area", 400, "security_error")
    
    # Get file/directory info for better logging
    file_type = "directory" if os.path.isdir(filepath) else "file" if os.path.isfile(filepath) else "unknown"
    file_name = os.path.basename(filepath)
    app.logger.info(f"[HF_DELETE_DRAFT] Deleting {file_type} '{file_name}'")
    
    # Call the delete function
    from models.huggingface import delete_draft_model
    result = delete_draft_model(filepath)
    app.logger.info(f"[HF_DELETE_DRAFT] Delete operation result: {result}")
    
    # Check result
    if not result.get("success"):
        error_msg = result.get('error', 'Unknown error')
        app.logger.error(f"[HF_DELETE_DRAFT] Delete operation failed: {error_msg}")
        return api_error_response(error_msg, 400, "delete_error", result)
    
    # Log success
    app.logger.info(f"[HF_DELETE_DRAFT] Successfully deleted {file_type} '{file_name}'")
    
    # Return the original result structure to maintain compatibility
    return result, 200

# API endpoint to update trending models limit
@app.route('/api/config/huggingface/search-limit', methods=['GET', 'POST'])
def update_search_results_limit() -> Tuple[Dict[str, Any], int]:
    """Get or update the search results limit configuration.
    
    This endpoint allows retrieving or changing the number of search results returned by the Hugging Face API.
    This limit applies to both trending models and regular search results.
    
    GET: Retrieve the current search results limit
    
    POST: Update the search results limit
    JSON parameters:
    - limit: The number of search results to return (integer, 1-100)
    
    Returns:
        tuple: API response with the current limit or update status
    """
    from models.config import get_setting, set_setting, save_settings
    
    # GET method: Return current limit
    if request.method == 'GET':
        limit = get_setting('frameworks.huggingface.search_results_limit', 50)
        return api_success_response(
            "Search results limit retrieved successfully",
            {"limit": limit}
        )
    
    # POST method: Update the limit
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    limit = data.get('limit')
    if limit is None:
        return api_error_response("Limit parameter is required", 400)
    
    try:
        limit = int(limit)
        if limit < 1 or limit > 100:
            return api_error_response("Limit must be between 1 and 100", 400)
    except ValueError:
        return api_error_response("Limit must be an integer", 400)
    
    app.logger.info(f"Updating search results limit to: {limit}")
    
    # Update the config
    success = set_setting('frameworks.huggingface.search_results_limit', limit)
    
    if not success:
        return api_error_response("Failed to update search results limit", 500)
    
    # Save the config
    if not save_settings():
        return api_error_response("Failed to save settings", 500)
    
    return api_success_response(f"Updated search results limit to {limit}", {"limit": limit})


@app.route('/api/config/huggingface/trending-search', methods=['GET', 'POST'])
def trending_search_config() -> Tuple[Dict[str, Any], int]:
    """Get or update the trending search configuration.
    
    This endpoint allows retrieving or updating the search parameters used for trending models.
    
    GET: Retrieve the current trending search configuration
    
    POST: Update the trending search configuration
    JSON parameters:
    - query: Optional search query for trending models (string)
    - model_type: Model type to filter by, e.g., "text-generation" (string)
    - tags: List of tags to filter by, e.g., ["gguf"] (array of strings)
    - sort_by: Field to sort by, e.g., "downloads", "last_modified" (string)
    - sort_direction: Sort direction: -1 for descending, 1 for ascending (integer)
    
    Returns:
        tuple: API response with configuration data and status code
    """
    from models.config import get_setting, set_setting, save_settings
    
    # GET method: Return current configuration
    if request.method == 'GET':
        trending_config = get_setting("frameworks.huggingface.trending_search", {
            "query": "",
            "model_type": "text-generation",
            "tags": ["gguf"],
            "sort_by": "last_modified",
            "sort_direction": -1
        })
        
        return api_success_response(
            "Trending search configuration retrieved successfully", 
            trending_config
        )
    
    # POST method: Update configuration
    data = request.json
    if not data:
        return api_error_response("No data provided", 400)
    
    # Get current config to update only provided fields
    current_config = get_setting("frameworks.huggingface.trending_search", {
        "query": "",
        "model_type": "text-generation",
        "tags": ["gguf"],
        "sort_by": "last_modified",
        "sort_direction": -1
    })
    
    # Update only the fields provided in the request
    if 'query' in data:
        current_config['query'] = data['query']
        
    if 'model_type' in data:
        current_config['model_type'] = data['model_type']
        
    if 'tags' in data:
        if not isinstance(data['tags'], list):
            return api_error_response("'tags' must be a list", 400)
        current_config['tags'] = data['tags']
        
    if 'sort_by' in data:
        valid_sort_fields = ['downloads', 'last_modified', 'likes']
        if data['sort_by'] not in valid_sort_fields:
            return api_error_response(f"'sort_by' must be one of: {', '.join(valid_sort_fields)}", 400)
        current_config['sort_by'] = data['sort_by']
        
    if 'sort_direction' in data:
        try:
            direction = int(data['sort_direction'])
            if direction not in [-1, 1]:
                return api_error_response("'sort_direction' must be either -1 or 1", 400)
            current_config['sort_direction'] = direction
        except (ValueError, TypeError):
            return api_error_response("'sort_direction' must be an integer (-1 or 1)", 400)
    
    app.logger.info(f"Updating trending search configuration: {current_config}")
    
    # Update the config
    success = set_setting("frameworks.huggingface.trending_search", current_config)
    if not success:
        return api_error_response("Failed to update trending search configuration", 500)
    
    # Save the config
    if not save_settings():
        return api_error_response("Failed to save settings", 500)
    
    return api_success_response(
        "Trending search configuration updated successfully",
        current_config
    )

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

def parse_arguments():
    """Parse command line arguments for the application.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description='LLM Model Manager')
    parser.add_argument('--port', type=int, default=config.get_port(), 
                        help=f'Port to run the server on (default: {config.get_port()})')
    parser.add_argument('--debug', action='store_true', default=config.get_debug_mode(), 
                        help='Run in debug mode')
    parser.add_argument('--model-library', type=str, default=config.get_model_library_path(),
                        help=f'Path to model library (default: {config.get_model_library_path()})')
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration based on command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    if args.port != config.get_port():
        config.set_setting('app.port', args.port)
    
    if args.debug != config.get_debug_mode():
        config.set_setting('app.debug', args.debug)
        
    if args.model_library != config.get_model_library_path():
        config.set_setting('paths.model_library', args.model_library)


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Update config with command line arguments
    update_config_from_args(args)
    
    # Run the application
    app.run(debug=args.debug, port=args.port)

