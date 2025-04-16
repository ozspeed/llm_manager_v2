"""
huggingface_controller.py (v2)
Flask Blueprint for Hugging Face API endpoints.
"""
from flask import Blueprint, jsonify, request
from models.search_history import get_popular_models
from llm_manager_v2.services.huggingface_service import search_models, get_trending_models

huggingface_bp = Blueprint('huggingface', __name__, url_prefix='/api/huggingface')

@huggingface_bp.route('/search', methods=['GET'])
def search_models_api():
    query = request.args.get('query', '')
    results = search_models(query)
    return jsonify({"results": results, "query": query})

@huggingface_bp.route('/trending', methods=['GET'])
def trending_models_api():
    results = get_trending_models()
    return jsonify({"results": results})

@huggingface_bp.route('/recent-searches', methods=['GET'])
def recent_searches():
    return jsonify(get_popular_models())
