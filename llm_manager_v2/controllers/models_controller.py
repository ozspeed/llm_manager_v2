"""
models_controller.py (v2)
Flask Blueprint for model CRUD API endpoints.
"""
from flask import Blueprint, jsonify, request, send_file
from services.model_service import list_models, add_model_service, get_model_details, delete_model_service

models_bp = Blueprint('models', __name__, url_prefix='/api/models')

@models_bp.route('', methods=['GET'])
def list_models_api():
    """List all models."""
    models, status = list_models()
    return jsonify(models), status

@models_bp.route('', methods=['POST'])
def upload_model_api():
    """Upload a new model (multipart form)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    name = request.form.get('name')
    framework = request.form.get('framework')
    if not name or not framework:
        return jsonify({'error': 'Name and framework required'}), 400
    result, status = add_model_service(name, framework, file)
    return jsonify(result), status

@models_bp.route('/<int:model_id>', methods=['GET'])
def get_model_api(model_id):
    """Get model details by ID."""
    result, status = get_model_details(model_id)
    return jsonify(result), status

@models_bp.route('/<int:model_id>', methods=['DELETE'])
def delete_model_api(model_id):
    """Delete a model by ID."""
    result, status = delete_model_service(model_id)
    return jsonify(result), status
