"""
settings_controller.py (v2)
Flask Blueprint for settings/config API endpoints.
"""
from flask import Blueprint, jsonify, request
from llm_manager_v2.models.settings_service import SettingsService

settings_bp = Blueprint('settings', __name__, url_prefix='/api/settings')
settings_service = SettingsService()

@settings_bp.route('', methods=['GET'])
def get_all_settings():
    all_settings = settings_service.all()
    return jsonify(all_settings), 200

@settings_bp.route('/<key>', methods=['GET'])
def get_setting(key):
    value = settings_service.get(key)
    if value is not None:
        return jsonify({"key": key, "value": value}), 200
    else:
        return jsonify({"error": f"Setting '{key}' not found."}), 404

@settings_bp.route('/<key>', methods=['PUT'])
def set_setting(key):
    data = request.get_json()
    value = data.get("value")
    if value is None:
        return jsonify({"error": "Missing value in request body."}), 400
    settings_service.set(key, value)
    return jsonify({"key": key, "value": value}), 200
