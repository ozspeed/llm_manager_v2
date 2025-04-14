"""
Database operations for the LLM Model Manager.

This module provides functionality for interacting with the SQLite database
that stores information about the models managed by the application.

It handles:
- Database initialization and schema creation
- Adding models to the database with metadata
- Retrieving models with filtering and sorting
- Deleting models with optional file removal
- Database reset and maintenance

The database schema includes fields for model name, framework, file path,
size, last used timestamp, and a JSON configuration field for additional metadata.
"""

# Standard library imports
import sqlite3
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# Application imports
from models import config

# =============================================================================
# DATABASE INITIALIZATION AND SCHEMA MANAGEMENT
# =============================================================================

def init_db() -> None:
    """Initialize the database and create tables if they don't exist.
    
    This function creates the SQLite database file if it doesn't exist
    and initializes the schema with the required tables. It creates a
    'models' table with fields for storing model metadata.
    
    The schema includes:
    - id: Unique identifier for each model
    - name: Display name of the model
    - framework: Framework type (e.g., 'gguf', 'safetensors', 'ollama')
    - path: File path to the model (must be unique)
    - size_mb: Size of the model in megabytes
    - last_used: Timestamp of when the model was last used
    - config: JSON string containing additional configuration data
    
    Returns:
        None
    """
    conn = sqlite3.connect(config.get_database_path())
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        framework TEXT NOT NULL,
        path TEXT NOT NULL UNIQUE,
        size_mb REAL,
        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config TEXT
    )
    ''')
    conn.commit()
    conn.close()

# =============================================================================
# MODEL RETRIEVAL AND QUERYING
# =============================================================================

def get_all_models() -> tuple[list[dict] | dict, int]:
    """Get all models from the database.
    
    This function retrieves all models from the database, ordered by name.
    It parses the JSON config field for each model and returns a list of
    model dictionaries with all metadata.
    
    Returns:
        tuple: A tuple containing:
            - list[dict] | dict: List of model dictionaries if successful or error dict
            - int: HTTP status code (200 for success, 500 for error)
            
    Each model dictionary contains the following keys:
    - id: Unique identifier for the model
    - name: Display name of the model
    - framework: Framework type
    - path: File path to the model
    - size_mb: Size of the model in megabytes
    - last_used: Timestamp of when the model was last used
    - config: Dictionary containing additional configuration data
    """
    try:
        conn = sqlite3.connect(config.get_database_path())
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY name")
        models = []
        for row in cursor.fetchall():
            model = dict(row)
            # Parse the config JSON if it exists
            if model['config']:
                try:
                    model['config'] = json.loads(model['config'])
                except json.JSONDecodeError:
                    model['config'] = {}
            else:
                model['config'] = {}
            models.append(model)
        conn.close()
        return models, 200
    except Exception as e:
        return {"error": str(e)}, 500

# =============================================================================
# MODEL CREATION AND MANAGEMENT
# =============================================================================

def add_model(name: str, framework: str, path: str, model_config: dict | str | None = None, 
           size_override: float | None = None) -> tuple[dict, int]:
    """Add a model to the database.
    
    This function adds a new model to the database with the provided metadata.
    It handles different types of models including regular files, virtual paths
    for Ollama models, and sharded models.
    
    Args:
        name (str): Display name of the model
        framework (str): Framework type (e.g., 'gguf', 'safetensors', 'ollama')
        path (str): File path to the model or virtual path for Ollama models
        model_config (dict or str, optional): Additional configuration data
        size_override (float, optional): Override for model size in MB
        
    Returns:
        tuple: A tuple containing:
            - dict: Response with model ID or error message
            - int: HTTP status code (201 for created, 400/500 for errors)
    """
    try:
        # Handle virtual paths for Ollama models or sharded models
        is_ollama_path = path.startswith('ollama://')
        
        # Check if model_config indicates this is a sharded model
        is_sharded = False
        if model_config is not None:
            if isinstance(model_config, dict):
                is_sharded = model_config.get('is_sharded', False)
            elif isinstance(model_config, str):
                try:
                    config_data = json.loads(model_config)
                    is_sharded = config_data.get('is_sharded', False)
                except:
                    pass
        
        if is_ollama_path:
            if size_override is None:
                return {"error": "Size must be provided for Ollama models"}, 400
            size_mb = size_override
            file_path = path  # Use the original path string for Ollama models
        elif is_sharded:
            # For sharded models, the path might be a virtual path that doesn't exist as a file
            file_path = path
            if size_override is None:
                return {"error": "Size must be provided for sharded models"}, 400
            size_mb = size_override
        else:
            # Regular file path validation
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File does not exist: {path}"}, 400
                
            # Use provided size or calculate from file
            size_mb = size_override if size_override is not None else file_path.stat().st_size / (1024 * 1024)
        
        conn = sqlite3.connect(config.get_database_path())
        cursor = conn.cursor()
        
        # Ensure model_config is a proper JSON string
        if model_config is not None:
            # If model_config is already a string, try to parse it to validate it's proper JSON
            if isinstance(model_config, str):
                try:
                    json.loads(model_config)  # Just to validate
                    config_json = model_config
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as a regular string
                    config_json = json.dumps({"raw_config": model_config})
            else:
                # If it's a dict or other object, convert to JSON string
                config_json = json.dumps(model_config)
        else:
            config_json = json.dumps({})
        
        # We've already determined if it's a sharded model above
        # Just ensure config_data is set for later use
        try:
            if not isinstance(config_data, dict):
                config_data = json.loads(config_json)
        except:
            config_data = {}
            
        # For sharded models, check if any model with the same name and framework exists
        if is_sharded:
            cursor.execute("SELECT id FROM models WHERE name = ? AND framework = ?", (name, framework))
            if cursor.fetchone():
                conn.close()
                return {"error": f"Model already exists with name: {name}"}, 400
        else:
            # Check if model already exists with the same path
            cursor.execute("SELECT id FROM models WHERE path = ?", (path,))
            if cursor.fetchone():
                conn.close()
                return {"error": f"Model already exists: {path}"}, 400
        
        # Insert new model
        cursor.execute(
            "INSERT INTO models (name, framework, path, size_mb, config) VALUES (?, ?, ?, ?, ?)",
            (name, framework, str(file_path), size_mb, config_json)
        )
        conn.commit()
        model_id = cursor.lastrowid
        conn.close()
        
        return {"id": model_id, "message": "Model added successfully"}, 201
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500

def delete_model(model_id: int, delete_files: bool = False) -> tuple[dict, int]:
    """Delete a model from the database by ID.
    
    This function removes a model from the database and optionally deletes
    the associated files from the filesystem. It handles both single file models
    and directory-based models with multiple files.
    
    Args:
        model_id (int): The ID of the model to delete
        delete_files (bool, optional): If True, also delete the underlying files
        
    Returns:
        tuple: A tuple containing:
            - dict: Response with success or error message
            - int: HTTP status code (200 for success, 404/500 for errors)
    """
    try:
        conn = sqlite3.connect(config.get_database_path())
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if model exists and get its path
        cursor.execute("SELECT id, path, config FROM models WHERE id = ?", (model_id,))
        model = cursor.fetchone()
        if not model:
            conn.close()
            return {"error": f"Model not found with ID: {model_id}"}, 404
        
        model_path = model['path']
        model_config = None
        if model['config']:
            try:
                model_config = json.loads(model['config'])
            except json.JSONDecodeError:
                model_config = {}
        
        # Delete the model from the database
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()
        
        # If requested, delete the underlying files
        if delete_files and model_path and not model_path.startswith('ollama://'):
            try:
                # Check if this is a model directory or a single file
                model_dir = None
                
                # If this is a model directory (contains model_dir in config)
                if model_config and 'model_dir' in model_config and 'publisher' in model_config:
                    # The path to the parent directory containing the model
                    publisher_name = model_config['publisher']
                    model_name = model_config['model_dir']
                    model_library = config.get_model_library_path()
                    model_dir = Path(os.path.join(model_library, publisher_name, model_name))
                else:
                    # This is a single file or we don't have enough info
                    file_path = Path(model_path)
                    if file_path.exists() and file_path.is_file():
                        # Delete the file
                        os.remove(file_path)
                        return {"message": f"Model {model_id} and its file deleted successfully"}, 200
                
                # If we found a model directory, delete it
                if model_dir and model_dir.exists() and model_dir.is_dir():
                    import shutil
                    shutil.rmtree(model_dir)
                    return {"message": f"Model {model_id} and its directory deleted successfully"}, 200
            except Exception as e:
                return {"message": f"Model {model_id} deleted from database, but error deleting files: {str(e)}"}, 200
        
        return {"message": f"Model {model_id} deleted successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

# =============================================================================
# DATABASE MAINTENANCE
# =============================================================================

def reset_database() -> tuple[dict, int]:
    """Reset the database by dropping and recreating the models table.
    
    This function completely resets the database by dropping the models table
    and recreating it with the original schema. This will delete all model
    entries from the database but will not delete any model files.
    
    Returns:
        tuple: A tuple containing:
            - dict: Response with success or error message
            - int: HTTP status code (200 for success, 500 for error)
    """
    try:
        conn = sqlite3.connect(config.get_database_path())
        cursor = conn.cursor()
        
        # Drop the models table
        cursor.execute("DROP TABLE IF EXISTS models")
        conn.commit()
        
        # Recreate the table
        cursor.execute('''
        CREATE TABLE models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            framework TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            size_mb REAL,
            last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config TEXT
        )
        ''')
        conn.commit()
        conn.close()
        
        return {"message": "Database reset successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500
