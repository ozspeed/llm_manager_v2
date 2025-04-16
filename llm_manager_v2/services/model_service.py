"""
model_service.py (v2)
Business logic for model management (CRUD, etc).
"""
from models.database import get_all_models
from typing import List, Dict, Tuple

def list_models() -> Tuple[List[Dict], int]:
    return get_all_models()
