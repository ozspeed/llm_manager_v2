"""
huggingface_controller.py (v2)
Flask Blueprint for Hugging Face API endpoints.
"""
from flask import Blueprint, jsonify, request
from models.search_history import get_popular_models

huggingface_bp = Blueprint('huggingface', __name__, url_prefix='/api/huggingface')

@huggingface_bp.route('/search', methods=['GET'])
def search_models():
    # TODO: Integrate with Hugging Face service/model
    query = request.args.get('query', '')
    # Placeholder response
    return jsonify({"results": [], "query": query})

@huggingface_bp.route('/recent-searches', methods=['GET'])
def recent_searches():
    return jsonify(get_popular_models())
