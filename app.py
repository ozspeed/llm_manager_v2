from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import psutil
import re
import subprocess
from pathlib import Path
import sqlite3
from datetime import datetime

# Configuration
APP_VERSION = "1.0.0"  # MVP 1.0
# Default paths - can be overridden by environment variables
MODEL_LIBRARY_PATH = os.environ.get('MODEL_LIBRARY_PATH', "/Volumes/Library_Bolt/AI Model Library")
DATABASE_PATH = os.environ.get('DATABASE_PATH', "models.db")
MODEL_EXTENSIONS = ['.gguf', '.ggml', '.bin', '.safetensors', '.onnx', '.pt', '.pth']

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Database setup
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        framework TEXT NOT NULL,
        path TEXT NOT NULL UNIQUE,
        size_mb REAL,
        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

def detect_framework(file_path):
    """Detect the framework based on file extension and name patterns"""
    path_str = str(file_path).lower()
    
    if path_str.endswith('.gguf') or path_str.endswith('.ggml'):
        return "llama.cpp"
    elif path_str.endswith('.bin') and ('pytorch' in path_str or 'torch' in path_str):
        return "transformers"
    elif path_str.endswith('.safetensors'):
        return "transformers"
    elif path_str.endswith('.onnx'):
        return "onnx"
    elif path_str.endswith('.pt') or path_str.endswith('.pth'):
        return "pytorch"
    elif 'ollama' in path_str and (path_str.endswith('manifest') or '/manifests/' in path_str):
        return "ollama"
    else:
        return "unknown"

def get_all_models():
    """Get all models from the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models ORDER BY last_used DESC")
    models = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Parse config JSON
    for model in models:
        try:
            model['config'] = json.loads(model['config']) if model['config'] else {}
        except:
            model['config'] = {}
    
    return models

def add_model(name, framework, path, config=None, size_override=None):
    """Add a model to the database"""
    try:
        # Handle virtual paths for Ollama models
        is_ollama_path = path.startswith('ollama://')
        
        if is_ollama_path:
            if size_override is None:
                return {"error": "Size must be provided for Ollama models"}, 400
            size_mb = size_override
            file_path = path  # Use the original path string for Ollama models
        else:
            # Regular file path validation
            file_path = Path(path)
            if not file_path.exists():
                return {"error": f"File does not exist: {path}"}, 400
                
            # Use provided size or calculate from file
            size_mb = size_override if size_override is not None else file_path.stat().st_size / (1024 * 1024)
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Ensure config is a proper JSON string
        if config is not None:
            # If config is already a string, try to parse it to validate it's proper JSON
            if isinstance(config, str):
                try:
                    json.loads(config)  # Just to validate
                    config_json = config
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as a regular string
                    config_json = json.dumps({"raw_config": config})
            else:
                # If it's a dict or other object, convert to JSON string
                config_json = json.dumps(config)
        else:
            config_json = json.dumps({})
        
        # Check if config indicates this is a sharded model
        is_sharded = False
        try:
            config_data = json.loads(config_json)
            is_sharded = config_data.get('is_sharded', False)
        except:
            pass
            
        # For sharded models, check if any model with the same name and framework exists
        if is_sharded:
            cursor.execute("SELECT id FROM models WHERE name = ? AND framework = ?", (name, framework))
            if cursor.fetchone():
                conn.close()
                return {"error": f"Model already exists with name: {name}"}, 400
        else:
            # Check if model already exists with the same path
            cursor.execute("SELECT id FROM models WHERE path = ?", (path,))
            if cursor.fetchone():
                conn.close()
                return {"error": f"Model already exists: {path}"}, 400
        
        # Insert new model
        cursor.execute(
            "INSERT INTO models (name, framework, path, size_mb, config) VALUES (?, ?, ?, ?, ?)",
            (name, framework, str(file_path), size_mb, config_json)
        )
        conn.commit()
        model_id = cursor.lastrowid
        conn.close()
        
        return {"id": model_id, "message": "Model added successfully"}, 201
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

def delete_model(model_id):
    """Delete a model from the database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            return {"error": f"Model not found: {model_id}"}, 404
            
        conn.commit()
        conn.close()
        return {"message": "Model deleted successfully"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def is_shard_file(filename):
    """Check if a file is a shard based on naming pattern"""
    # Focus on the most common shard patterns
    patterns = [
        r'.*-\d{5}-of-\d{5}\.',  # matches: name-00001-of-00005.ext
        r'.*\.\d{5}\.',           # matches: name.00001.ext
        r'.*-part\d+\.',          # matches: name-part1.ext
        r'.*-shard\d+\.'          # matches: name-shard1.ext
    ]
    
    # Check if the filename matches any pattern
    for pattern in patterns:
        if re.search(pattern, filename):
            return True
    return False

def extract_base_name(filename):
    """Extract base name from a sharded file"""
    # Remove extension first
    base = os.path.splitext(filename)[0]
    
    # Try different patterns to extract the base name
    patterns = [
        r'(.+?)-\d{5}-of-\d{5}$',  # matches: name-00001-of-00005
        r'(.+?)\.\d{5}$',          # matches: name.00001
        r'(.+?)-part\d+$',          # matches: name-part1
        r'(.+?)-shard\d+$'          # matches: name-shard1
    ]
    
    for pattern in patterns:
        match = re.search(pattern, base)
        if match:
            return match.group(1)
    
    return base

def count_shards(base_name, file_list, extension):
    """Count how many shards exist for a base name"""
    count = 0
    matching_files = []
    
    for file in file_list:
        file_ext = os.path.splitext(file)[1]
        if file_ext == extension:
            extracted_base = extract_base_name(file)
            print(f"  Comparing '{extracted_base}' with '{base_name}'")
            if extracted_base == base_name and is_shard_file(file):
                count += 1
                matching_files.append(file)
    
    print(f"Found {count} shards for base name '{base_name}': {matching_files}")
    return count

def scan_directory(directory_path):
    """Scan directory for model files and add them to the database"""
    try:
        directory = Path(directory_path)
        if not directory.exists():
            return {"error": f"Directory does not exist: {directory_path}"}, 400
        
        added_models = []
        processed_files = set()  # Keep track of processed files
        
        # First pass: collect all files by directory
        files_by_dir = {}
        for root, _, files in os.walk(directory):
            files_by_dir[root] = files
        
        # Process each directory
        for root, files in files_by_dir.items():
            # Find all model files in this directory
            model_files = [f for f in files if any(f.lower().endswith(ext) for ext in MODEL_EXTENSIONS)]
            
            # Group files by their base name
            shard_groups = {}
            for file in model_files:
                if is_shard_file(file):
                    base_name = extract_base_name(file)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file)
            
            # Process shard groups first (files with the same base name)
            for base_name, group_files in shard_groups.items():
                if len(group_files) > 1:  # Only process as a group if there are multiple files
                    # Calculate total size
                    total_size = 0
                    shard_paths = []
                    
                    for shard_file in group_files:
                        file_path = Path(root) / shard_file
                        if str(file_path) in processed_files:
                            continue  # Skip if already processed
                            
                        total_size += file_path.stat().st_size
                        shard_paths.append(str(file_path))
                        processed_files.add(str(file_path))  # Mark as processed
                    
                    if not shard_paths:  # Skip if all files were already processed
                        continue
                        
                    # Create model entry for the shard group
                    name = base_name.replace('_', ' ').replace('-', ' ')
                    framework = detect_framework(Path(shard_paths[0]))
                    
                    # Store shard info in config
                    config = json.dumps({
                        "is_sharded": True,
                        "shard_count": len(shard_paths),
                        "shard_paths": shard_paths
                    })
                    
                    # Add model to database with the first shard as the main path
                    result, status_code = add_model(
                        name, 
                        framework, 
                        shard_paths[0],  # Use first shard as reference path
                        config=config,
                        size_override=total_size / (1024 * 1024)  # Convert to MB
                    )
                    
                    # Only add to results if it was added successfully
                    if status_code == 201:
                        added_models.append({
                            "name": name,
                            "framework": framework,
                            "path": shard_paths[0],
                            "is_sharded": True,
                            "shard_count": len(shard_paths)
                        })
            
            # Process remaining files that aren't part of shard groups
            for file in model_files:
                file_path = Path(root) / file
                
                # Skip if already processed
                if str(file_path) in processed_files:
                    continue
                
                # Skip if this is a shard file that should be part of a group
                if is_shard_file(file) and extract_base_name(file) in shard_groups:
                    continue
                
                # Create model name from filename
                name = os.path.splitext(file)[0].replace('_', ' ').replace('-', ' ')
                
                # Detect framework
                framework = detect_framework(file_path)
                
                # Add model to database
                result, status_code = add_model(name, framework, str(file_path))
                processed_files.add(str(file_path))  # Mark as processed
                
                # Only add to results if it was added successfully
                if status_code == 201:
                    added_models.append({
                        "name": name,
                        "framework": framework,
                        "path": str(file_path)
                    })
        
        return {"added": len(added_models), "models": added_models}, 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

# Routes
@app.route('/')
def index():
    return render_template('index_new.html')
    
@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(get_all_models())

@app.route('/api/models', methods=['POST'])
def create_model():
    try:
        data = request.json
        name = data.get('name')
        framework = data.get('framework')
        path = data.get('path')
        config = data.get('config', {})
        
        if not all([name, framework, path]):
            return jsonify({"error": "Missing required fields"}), 400
            
        result, status_code = add_model(name, framework, path, config)
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def remove_model(model_id):
    result, status_code = delete_model(model_id)
    return jsonify(result), status_code

@app.route('/api/scan', methods=['GET'])
def scan_models():
    path = request.args.get('path', MODEL_LIBRARY_PATH)
    include_ollama = request.args.get('include_ollama', 'true').lower() == 'true'
    
    # Scan filesystem models
    result, status_code = scan_directory(path)
    
    # Scan Ollama models if requested
    if include_ollama:
        ollama_result = scan_ollama_models()
        
        # Update the result with Ollama models
        if status_code == 200 and ollama_result and 'models' in ollama_result:
            result['added'] += len(ollama_result['models'])
            result['models'].extend(ollama_result['models'])
            result['ollama_models_found'] = len(ollama_result['models'])
    
    return jsonify(result), status_code

@app.route('/api/system-info', methods=['GET'])
def system_info():
    # System memory info
    memory = psutil.virtual_memory()._asdict()
    
    # Root disk info
    root_disk = psutil.disk_usage('/')._asdict()
    
    # Model library disk info
    library_disk = None
    if MODEL_LIBRARY_PATH and os.path.exists(MODEL_LIBRARY_PATH):
        try:
            library_disk = psutil.disk_usage(MODEL_LIBRARY_PATH)._asdict()
        except Exception as e:
            print(f"Error getting disk usage for {MODEL_LIBRARY_PATH}: {e}")
            pass
    
    # Get total size of all models
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(size_mb) FROM models")
    total_model_size = cursor.fetchone()[0] or 0
    conn.close()
    
    return jsonify({
        "memory": memory,
        "root_disk": root_disk,
        "library_disk": library_disk,
        "total_model_size_mb": total_model_size,
        "model_count": len(get_all_models())
    })

@app.route('/api/reset-database', methods=['POST'])
def reset_database():
    """Reset the database - CAUTION: This will delete all models"""
    try:
        # Delete all models from the database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM models")
        conn.commit()
        conn.close()
        return jsonify({"message": "Database reset successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def parse_ollama_modelfile(modelfile_content):
    """Parse Ollama Modelfile content to extract metadata"""
    metadata = {
        "parameters": {},
        "system_prompt": None,
        "template": None,
        "license": None,
        "from_model": None,
        "display_name": None  # Added for human-readable name
    }
    
    lines = modelfile_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Extract FROM directive
        if line.startswith('FROM '):
            metadata['from_model'] = line[5:].strip()
        
        # Extract PARAMETER directives
        elif line.startswith('PARAMETER '):
            parts = line[10:].strip().split(' ', 1)
            if len(parts) == 2:
                param_name, param_value = parts
                try:
                    # Try to convert to appropriate type
                    if param_value.replace('.', '', 1).isdigit():
                        if '.' in param_value:
                            param_value = float(param_value)
                        else:
                            param_value = int(param_value)
                    # Remove quotes if present
                    elif param_value.startswith('"') and param_value.endswith('"'):
                        param_value = param_value[1:-1]
                except ValueError:
                    pass  # Keep as string if conversion fails
                    
                metadata['parameters'][param_name] = param_value
        
        # Extract SYSTEM prompt
        elif line.startswith('SYSTEM '):
            metadata['system_prompt'] = line[7:].strip()
            # Remove quotes if present
            if metadata['system_prompt'].startswith('"') and metadata['system_prompt'].endswith('"'):
                metadata['system_prompt'] = metadata['system_prompt'][1:-1]
                
        # Extract TEMPLATE
        elif line.startswith('TEMPLATE '):
            # Templates can be multi-line, just capture the first line for now
            template_start = line[9:].strip()
            metadata['template'] = template_start
            
        # Extract LICENSE
        elif line.startswith('LICENSE '):
            # License can be multi-line, just capture the first line for now
            license_start = line[8:].strip()
            metadata['license'] = license_start
    
    return metadata

def scan_ollama_models():
    """Scan for Ollama models and add them to the database"""
    try:
        print("\n==== Starting Ollama model scan ====")
        # Check if Ollama is installed
        try:
            result = subprocess.run(['which', 'ollama'], capture_output=True, text=True, check=False)
            print(f"Ollama check result: {result.stdout.strip()}")
            if result.returncode != 0:
                print("Ollama not found in PATH")
                return {"models": [], "error": "Ollama not installed"}
        except Exception as e:
            print(f"Error checking for Ollama: {e}")
            return {"models": [], "error": f"Error checking for Ollama: {e}"}
        
        # Get list of Ollama models
        try:
            print("Running 'ollama list' command...")
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"Error running 'ollama list': {result.stderr}")
                return {"models": [], "error": f"Error running 'ollama list': {result.stderr}"}
                
            output = result.stdout
            print(f"Ollama list output:\n{output}")
        except Exception as e:
            print(f"Error running Ollama command: {e}")
            return {"models": [], "error": f"Error running Ollama command: {e}"}
        
        # Parse the output to get model names
        lines = output.strip().split('\n')
        print(f"Found {len(lines)} lines in Ollama output")
        if len(lines) <= 1:  # Only header or empty
            print("No Ollama models found (only header line)")
            return {"models": []}
            
        added_models = []
        
        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:  # NAME ID SIZE ...
                model_name = parts[0].strip()  # This is the model name with tag (e.g., 'deepseek-r1:70b')
                model_id = parts[1].strip()  # This is the hash ID
                
                # Extract size in MB
                size_str = parts[2].strip()
                size_mb = 0
                
                if size_str.endswith('GB'):
                    size_mb = float(size_str[:-2]) * 1024  # Convert GB to MB
                elif size_str.endswith('MB'):
                    size_mb = float(size_str[:-2])
                
                # Get model details using ollama show
                try:
                    print(f"Getting modelfile for {model_name}...")
                    modelfile_result = subprocess.run(
                        ['ollama', 'show', '--modelfile', model_name], 
                        capture_output=True, 
                        text=True, 
                        check=False
                    )
                    
                    if modelfile_result.returncode == 0:
                        modelfile_content = modelfile_result.stdout
                        print(f"Got modelfile for {model_name}, parsing...")
                        metadata = parse_ollama_modelfile(modelfile_content)
                        print(f"Parsed metadata: {metadata}")
                    else:
                        metadata = {}
                        print(f"Error getting modelfile for {model_name}: {modelfile_result.stderr}")
                except Exception as e:
                    metadata = {}
                    print(f"Error getting modelfile details: {e}")
                
                # Create a path for the Ollama model
                # This is a virtual path since Ollama manages its own storage
                ollama_path = f"ollama://{model_name}"
                
                # Extract a more human-readable display name from the model name
                # If it has a tag (e.g., 'deepseek-r1:70b'), use that as the display name
                display_name = model_name
                if ':' in model_name:
                    model_parts = model_name.split(':')
                    base_name = model_parts[0]
                    tag = model_parts[1]
                    # Convert to title case for better readability
                    display_name = f"{base_name.replace('-', ' ').title()} {tag}"
                else:
                    # If no tag, just format the name nicely
                    display_name = model_name.replace('-', ' ').title()
                
                # Store the display name in metadata
                metadata["display_name"] = display_name
                
                # Create config with Ollama-specific details
                config = {
                    "ollama_id": model_id,
                    "modelfile": metadata,
                    "is_ollama": True,
                    "display_name": display_name
                }
                
                # Add model to database - use the display name instead of the raw model name
                print(f"Adding model to database: {display_name} (path: {ollama_path})")
                result, status_code = add_model(
                    display_name,  # Use the human-readable display name
                    "ollama", 
                    ollama_path,
                    config=config,
                    size_override=size_mb
                )
                print(f"Add model result: {result}, status code: {status_code}")
                
                # Only add to results if it was added successfully
                if status_code == 201:
                    added_models.append({
                        "name": display_name,  # Use the human-readable display name
                        "original_name": model_name,  # Keep the original name for reference
                        "framework": "ollama",
                        "path": ollama_path,
                        "size_mb": size_mb,
                        "ollama_id": model_id,
                        "metadata": metadata
                    })
                elif status_code == 400 and "already exists" in str(result.get("error", "")):
                    # Model already exists, but we'll include it in the response anyway
                    added_models.append({
                        "name": display_name,  # Use the human-readable display name
                        "original_name": model_name,  # Keep the original name for reference
                        "framework": "ollama",
                        "path": ollama_path,
                        "size_mb": size_mb,
                        "ollama_id": model_id,
                        "metadata": metadata,
                        "already_exists": True
                    })
        
        print(f"Successfully processed {len(added_models)} Ollama models")
        return {"models": added_models}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Exception in scan_ollama_models: {e}")
        return {"models": [], "error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)
