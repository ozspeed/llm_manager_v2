from flask import Blueprint, jsonify
from llm_manager_v2.services.library_scan_service import LibraryScanService
from llm_manager_v2.models.settings_service import SettingsService

library_scan_bp = Blueprint('library_scan', __name__, url_prefix='/api/models')
settings_service = SettingsService()
library_scan_service = LibraryScanService(settings_service)

@library_scan_bp.route('/scan', methods=['POST'])
def scan_library():
    try:
        model_records = library_scan_service.scan_library_root()
        return jsonify([record.to_dict() for record in model_records]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
