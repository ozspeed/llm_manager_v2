"""
Database operations for the LLM Model Manager.
Handles model storage, retrieval, and management in the SQLite database.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import traceback

from config import DATABASE_PATH

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DATABASE_PATH)
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

def get_all_models():
    """Get all models from the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
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

def add_model(name, framework, path, config=None, size_override=None):
    """Add a model to the database."""
    try:
        # Handle virtual paths for Ollama models or sharded models
        is_ollama_path = path.startswith('ollama://')
        
        # Check if config indicates this is a sharded model
        is_sharded = False
        if config is not None:
            if isinstance(config, dict):
                is_sharded = config.get('is_sharded', False)
            elif isinstance(config, str):
                try:
                    config_data = json.loads(config)
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
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Ensure config is a proper JSON string
        if config is not None:
            # If config is already a string, try to parse it to validate it's proper JSON
            if isinstance(config, str):
                try:
                    json.loads(config)  # Just to validate
                    config_json = config
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as a regular string
                    config_json = json.dumps({"raw_config": config})
            else:
                # If it's a dict or other object, convert to JSON string
                config_json = json.dumps(config)
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

def delete_model(model_id):
    """Delete a model from the database by ID."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if model exists
        cursor.execute("SELECT id FROM models WHERE id = ?", (model_id,))
        if not cursor.fetchone():
            conn.close()
            return {"error": f"Model not found with ID: {model_id}"}, 404
        
        # Delete the model
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()
        
        return {"message": f"Model {model_id} deleted successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def reset_database():
    """Reset the database by dropping and recreating the models table."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
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
