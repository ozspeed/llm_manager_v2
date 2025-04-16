"""
settings_controller.py (v2)
Flask Blueprint for settings/config API endpoints.
"""
from flask import Blueprint, jsonify
from config.settings import get_config

settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')

@settings_bp.route('', methods=['GET'])
def get_settings():
    config = get_config()
    return jsonify(config.as_dict())
