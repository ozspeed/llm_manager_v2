"""
model_service.py (v2)
Business logic for model management (CRUD, etc).
"""
from models.database import get_all_models, add_model, delete_model
from typing import List, Dict, Tuple
from pathlib import Path
import tempfile
import os
import json

def list_models() -> Tuple[List[Dict], int]:
    return get_all_models()

def add_model_service(name: str, framework: str, file_storage) -> Tuple[Dict, int]:
    # Save the uploaded file to a models/ directory (create if needed)
    models_dir = Path('models_uploads')
    models_dir.mkdir(exist_ok=True)
    dest_path = models_dir / file_storage.filename
    file_storage.save(dest_path)
    # Add to database
    result, status = add_model(name, framework, str(dest_path))
    return result, status

def get_model_details(model_id: int) -> Tuple[Dict, int]:
    models, code = get_all_models()
    if code != 200:
        return {"error": "Could not fetch models"}, 500
    for m in models:
        if m.get('id') == model_id:
            return m, 200
    return {"error": f"Model {model_id} not found"}, 404

def delete_model_service(model_id: int) -> Tuple[Dict, int]:
    # Remove from database and optionally delete file
    result, status = delete_model(model_id, delete_files=True)
    return result, status
