"""
models_controller.py (v2)
Flask Blueprint for model CRUD API endpoints.
"""
from flask import Blueprint, jsonify, request
from models.database import get_all_models

models_bp = Blueprint('models', __name__, url_prefix='/api/models')

@models_bp.route('', methods=['GET'])
def list_models():
    models, status = get_all_models()
    return jsonify(models), status
