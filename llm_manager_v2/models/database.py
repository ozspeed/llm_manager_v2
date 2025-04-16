"""
database.py (v2)

Database operations for the LLM Model Manager v2.
- Handles database initialization, model CRUD, and schema management.
- All config loaded via config.settings.get_config().
- Type annotations and error handling throughout.
"""
import sqlite3
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import traceback

from llm_manager_v2.config.settings import get_config

# =============================================================================
# DATABASE INITIALIZATION AND SCHEMA MANAGEMENT
# =============================================================================
def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    config = get_config()
    db_path = getattr(config, 'DB_PATH', 'models.db')
    conn = sqlite3.connect(db_path)
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

def add_model(name: str, framework: str, path: str, model_config: Optional[dict] = None, size_override: Optional[float] = None) -> Tuple[Dict, int]:
    """Add a model to the database."""
    config = get_config()
    db_path = getattr(config, 'DB_PATH', 'models.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        file_path = Path(path)
        size_mb = size_override or (file_path.stat().st_size / (1024 ** 2) if file_path.exists() else 0)
        config_json = json.dumps(model_config) if model_config else '{}'
        # Check if model already exists
        cursor.execute("SELECT id FROM models WHERE path = ?", (str(file_path),))
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

def get_all_models() -> Tuple[List[Dict], int]:
    """Get all models from the database."""
    config = get_config()
    db_path = getattr(config, 'DB_PATH', 'models.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY name")
        models = []
        for row in cursor.fetchall():
            model = dict(zip([column[0] for column in cursor.description], row))
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

def delete_model(model_id: int, delete_files: bool = False) -> Tuple[Dict, int]:
    """Delete a model from the database by ID."""
    config = get_config()
    db_path = getattr(config, 'DB_PATH', 'models.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM models WHERE id = ?", (model_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {"error": f"Model {model_id} not found"}, 404
        model_path = Path(row[0])
        # Delete from database
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
        conn.close()
        # Optionally delete files
        if delete_files and model_path.exists():
            try:
                if model_path.is_file():
                    model_path.unlink()
                elif model_path.is_dir():
                    shutil.rmtree(model_path)
            except Exception as e:
                return {"message": f"Model {model_id} deleted from database, but error deleting files: {str(e)}"}, 200
        return {"message": f"Model {model_id} deleted successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def reset_database() -> Tuple[Dict, int]:
    """Reset the database by dropping and recreating the models table."""
    config = get_config()
    db_path = getattr(config, 'DB_PATH', 'models.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
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
