"""Main Flask application for the LLM Model Manager.

This module serves as the entry point for the application and defines all the routes.
It imports functionality from other modules to keep the code modular and maintainable.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Import configuration
import config

# Import modules
from models.database import init_db, get_all_models, add_model, delete_model, reset_database
from models.detection import scan_directory
from models.ollama import scan_ollama_models
from utils.system import get_system_info
from utils.file_browser import browse_directories

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize database
init_db()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('index_new.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    return get_all_models()

@app.route('/api/models', methods=['POST'])
def create_model():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    name = data.get('name')
    framework = data.get('framework')
    path = data.get('path')
    config = data.get('config')
    
    if not name or not framework or not path:
        return jsonify({"error": "Missing required fields: name, framework, path"}), 400
    
    return add_model(name, framework, path, config)

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def remove_model(model_id):
    return delete_model(model_id)

@app.route('/api/scan', methods=['GET'])
def scan_models():
    path = request.args.get('path', config.get_model_library_path())
    include_ollama = request.args.get('include_ollama', str(config.get_ollama_enabled())).lower() == 'true'
    
    # Scan filesystem models
    result, status_code = scan_directory(path)
    
    # Scan Ollama models if requested and enabled in config
    if include_ollama and config.get_ollama_enabled():
        ollama_result = scan_ollama_models(repositories=config.get_ollama_repositories())
        
        # Update the result with Ollama models
        if status_code == 200 and ollama_result and 'models' in ollama_result:
            # Count existing Ollama models
            existing_ollama = sum(1 for model in ollama_result['models'] if model.get('already_exists', False))
            new_ollama = len(ollama_result['models']) - existing_ollama
            
            result['added'] += new_ollama
            result['existing'] = result.get('existing', 0) + existing_ollama
            result['models'].extend(ollama_result['models'])
            result['ollama_models_found'] = len(ollama_result['models'])
    
    return jsonify(result), status_code

@app.route('/api/system-info', methods=['GET'])
def system_info():
    info, status_code = get_system_info()
    return jsonify(info), status_code

@app.route('/api/reset-database', methods=['POST'])
def reset_db():
    return reset_database()

# API endpoint to get configuration
@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the current configuration."""
    return jsonify(config.config)

# API endpoint to browse directories
@app.route('/api/browse', methods=['GET'])
def browse_dir():
    """Browse directories for the directory selector."""
    path = request.args.get('path', '/')
    return browse_directories(path)

# API endpoint to update configuration
@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration settings."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        # Update each setting provided
        for path, value in data.items():
            config.set_setting(path, value)
        
        return jsonify({"success": True, "message": "Configuration updated", "config": config.config})
    except Exception as e:
        return jsonify({"error": f"Failed to update configuration: {str(e)}"}), 500

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLM Model Manager')
    parser.add_argument('--port', type=int, default=config.get_port(), 
                        help=f'Port to run the server on (default: {config.get_port()})')
    parser.add_argument('--debug', action='store_true', default=config.get_debug_mode(), 
                        help='Run in debug mode')
    parser.add_argument('--model-library', type=str, default=config.get_model_library_path(),
                        help=f'Path to model library (default: {config.get_model_library_path()})')
    
    args = parser.parse_args()
    
    # Update config with command line arguments if they differ from defaults
    if args.port != config.get_port():
        config.set_setting('app.port', args.port)
    
    if args.debug != config.get_debug_mode():
        config.set_setting('app.debug', args.debug)
        
    if args.model_library != config.get_model_library_path():
        config.set_setting('paths.model_library', args.model_library)
    
    # Run the application
    app.run(debug=args.debug, port=args.port)
