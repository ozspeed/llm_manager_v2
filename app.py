"""Main Flask application for the LLM Model Manager.

This module serves as the entry point for the application and defines all the routes.
It imports functionality from other modules to keep the code modular and maintainable.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Import configuration
from models import config
import os
import shutil
import glob
import re
import sqlite3
import json
import datetime

# Import modules
from models.database import init_db, get_all_models, add_model, delete_model, reset_database
from models.detection import scan_directory, is_shard_file, extract_base_name
from models.ollama import scan_ollama_models

from utils.system import get_system_info
from utils.file_browser import browse_directories
from utils.model_scanner import scan_for_models

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize database
init_db()

# Routes
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/settings')
def settings():
    return render_template('settings.html')

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
    # Check if we should delete the files as well
    delete_files = request.args.get('delete_files', 'false').lower() == 'true'
    return delete_model(model_id, delete_files)

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

@app.route('/api/refresh-models', methods=['POST'])
def refresh_models():
    """Refresh the models in the database by scanning the model library."""
    try:
        # Get the model library path
        model_library = config.get_model_library_path()
        
        # Scan the model library
        result, status_code = scan_directory(model_library)
        
        # Include Ollama models if enabled
        if config.get_ollama_enabled():
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset-database', methods=['POST'])
def reset_db():
    return reset_database()

# API endpoint to add a model with directory structure
@app.route('/api/add-model', methods=['POST'])
def add_model_with_directory():
    """Add a model by copying files from source to the model library with directory structure."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    source_dir = data.get('sourceDir')
    model_dir = data.get('modelDir')
    model_name = data.get('modelName')
    
    # We'll auto-detect the framework and always handle sharded models
    framework = data.get('framework', 'unknown')
    handle_shards = True
    
    if not source_dir or not model_dir or not model_name:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Check if source directory exists
    if not os.path.isdir(model_dir):
        return jsonify({"error": f"Model directory does not exist: {model_dir}"}), 400
    
    # Create publisher directory in model library if it doesn't exist
    publisher_name = os.path.basename(source_dir)
    publisher_dir = os.path.join(model_library, publisher_name)
    
    if not os.path.exists(publisher_dir):
        os.makedirs(publisher_dir)
    
    # Create model directory path in the library
    target_model_dir = os.path.join(publisher_dir, model_name)
    
    # Check if model directory already exists in the library
    if os.path.exists(target_model_dir):
        return jsonify({
            "error": f"Model directory already exists: {target_model_dir}",
            "needsConfirmation": True,
            "modelName": model_name,
            "modelPath": target_model_dir
        }), 409
    
    try:
        # Create the model directory in the library
        os.makedirs(target_model_dir)
        
        # Find model files in the source directory
        model_extensions = config.get_model_extensions()
        model_files = []
        
        for ext in model_extensions:
            pattern = os.path.join(model_dir, f"*.{ext}")
            model_files.extend(glob.glob(pattern))
        
        if not model_files:
            # Clean up the empty directory we created
            os.rmdir(target_model_dir)
            return jsonify({"error": f"No model files found in {model_dir}"}), 400
        
        # Process sharded models if enabled
        if handle_shards:
            # Group files by their base name (for sharded models)
            shard_groups = {}
            non_sharded_files = []
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if is_shard_file(filename):
                    base_name = extract_base_name(filename)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file_path)
                else:
                    non_sharded_files.append(file_path)
            
            # Copy each group of sharded files
            for base_name, shard_files in shard_groups.items():
                for shard_file in shard_files:
                    filename = os.path.basename(shard_file)
                    target_path = os.path.join(target_model_dir, filename)
                    print(f"Copying shard file: {shard_file} -> {target_path}")
                    shutil.copy2(shard_file, target_path)
            
            # Copy non-sharded files
            for file_path in non_sharded_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
        else:
            # Copy all files without shard processing
            for file_path in model_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
        
        # Add the model to the database
        # For sharded models, use the virtual path to the directory
        if handle_shards and shard_groups:
            # Add each sharded model group to the database
            for base_name, shard_files in shard_groups.items():
                # Calculate total size of all shards
                total_size_mb = sum(os.path.getsize(f) for f in shard_files) / (1024 * 1024)
                
                # Create a friendly name from the base name
                friendly_name = f"{model_name} - {base_name}"
                
                # Create a virtual path to the model directory
                virtual_path = os.path.join(target_model_dir, base_name)
                
                # Add to database with shard information
                result, status_code = add_model(
                    name=friendly_name,
                    framework=framework,
                    path=virtual_path,
                    model_config={
                        "is_sharded": True,
                        "shard_count": len(shard_files),
                        "publisher": publisher_name,
                        "model_dir": model_name
                    },
                    size_override=total_size_mb
                )
                
                if status_code != 201:
                    return jsonify({
                        "error": f"Failed to add sharded model to database: {result.get('error', 'Unknown error')}"
                    }), status_code
        
        # Add non-sharded models to the database
        for file_path in non_sharded_files if handle_shards else model_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_model_dir, filename)
            
            # Calculate file size
            size_mb = os.path.getsize(target_path) / (1024 * 1024)
            
            # Add to database
            result, status_code = add_model(
                name=f"{model_name} - {filename}",
                framework=framework,
                path=target_path,
                model_config={
                    "is_sharded": False,
                    "publisher": publisher_name,
                    "model_dir": model_name
                },
                size_override=size_mb
            )
            
            if status_code != 201:
                return jsonify({
                    "error": f"Failed to add model to database: {result.get('error', 'Unknown error')}"
                }), status_code
        
        return jsonify({
            "message": f"Successfully added model(s) from {model_name}",
            "modelPath": target_model_dir
        }), 201
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(target_model_dir):
            shutil.rmtree(target_model_dir)
        return jsonify({"error": str(e)}), 500

# API endpoint to confirm model overwrite
@app.route('/api/confirm-overwrite', methods=['POST'])
def confirm_overwrite():
    """Confirm overwriting an existing model."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    source_dir = data.get('sourceDir')
    model_dir = data.get('modelDir')
    model_name = data.get('modelName')
    framework = data.get('framework')
    handle_shards = data.get('handleShards', True)
    confirm = data.get('confirm', False)
    
    if not confirm:
        return jsonify({"message": "Operation cancelled"}), 200
    
    if not source_dir or not model_dir or not model_name or not framework:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Determine the target model directory
    publisher_name = os.path.basename(source_dir)
    publisher_dir = os.path.join(model_library, publisher_name)
    target_model_dir = os.path.join(publisher_dir, model_name)
    
    try:
        # Find all models in the database with this path prefix
        conn = sqlite3.connect(config.get_database_path())
        cursor = conn.cursor()
        
        # Use LIKE with path prefix to find all related models
        cursor.execute("SELECT id FROM models WHERE path LIKE ?", (f"{target_model_dir}%",))
        model_ids = [row[0] for row in cursor.fetchall()]
        
        # Delete models from database
        for model_id in model_ids:
            delete_model(model_id)
        
        # Delete the directory and its contents
        if os.path.exists(target_model_dir):
            shutil.rmtree(target_model_dir)
        
        # Now proceed with adding the model again
        return add_model_with_directory()
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint to get configuration
@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the current configuration."""
    if config._settings is None:
        config.load_settings()
    return jsonify(config._settings)

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
        # Update all settings at once
        if config.update_settings(data):
            return jsonify({
                "success": True, 
                "message": "Configuration updated", 
                "config": config._settings
            })
        else:
            return jsonify({"error": "Failed to update configuration"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to update configuration: {str(e)}"}), 500

# API endpoint to sync repository with AI Library
@app.route('/api/sync_repository', methods=['POST'])
def sync_repository():
    """Sync repository with AI Library by creating symlinks based on repository settings."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    name = data.get('name')
    location = data.get('location')
    require_parent_directory = data.get('require_parent_directory', False)
    understands_shards = data.get('understands_shards', False)
    
    if not name or not location:
        return jsonify({"error": "Repository name and location are required"}), 400
    
    try:
        # Get the model library path
        model_library_path = config.get_model_library_path()
        
        if not model_library_path:
            return jsonify({"error": "Model library path is not set"}), 400
        
        # Import necessary modules
        import os
        import shutil
        import datetime
        import re
        from pathlib import Path
        
        # Create symlinks from model library to repository location
        model_count = 0
        skipped_count = 0
        sharded_count = 0
        model_extensions = config.get_model_extensions()
        
        # Ensure repository directory exists
        os.makedirs(location, exist_ok=True)
        
        # Track models by type
        model_types = {}
        
        # First, prescan the target directory to find existing symlinks
        existing_symlinks = {}
        for root, dirs, files in os.walk(location):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.islink(file_path):
                    # Get the real path that the symlink points to
                    try:
                        real_path = os.path.realpath(file_path)
                        # Store the real path as a key to avoid duplicate symlinks
                        existing_symlinks[real_path] = file_path
                    except Exception:
                        # If we can't resolve the symlink, just continue
                        pass
        
        # Scan model library for models
        for ext in model_extensions:
            for model_file in Path(model_library_path).glob(f'**/*{ext}'):
                # Skip sharded models if the repository doesn't understand them
                # More robust shard detection - looking for patterns like .00.gguf, .01.gguf, etc.
                is_shard = False
                file_name = model_file.name
                
                # Check for common shard patterns
                # Pattern 1: name.00.ext, name.01.ext, etc.
                if re.search(r'\.[0-9]{2,}\.[^.]+$', file_name):
                    is_shard = True
                # Pattern 2: name-00-of-03.ext, name-01-of-03.ext, etc.
                elif re.search(r'-[0-9]{2,}-of-[0-9]{2,}\.[^.]+$', file_name):
                    is_shard = True
                # Pattern 3: name.shard0.ext, name.shard1.ext, etc.
                elif re.search(r'\.shard[0-9]+\.[^.]+$', file_name):
                    is_shard = True
                
                if is_shard and not understands_shards:
                    sharded_count += 1
                    continue
                
                # Check if this model is already symlinked somewhere in the target directory
                real_model_path = str(model_file.resolve())
                if real_model_path in existing_symlinks:
                    skipped_count += 1
                    continue
                
                # Determine the target path based on whether parent directories are required
                if require_parent_directory:
                    # Get the relative path from the model library to the model file's parent
                    rel_path = os.path.relpath(model_file.parent, model_library_path)
                    if rel_path == ".":
                        # If the model is directly in the model library, just use the repository location
                        target_dir = location
                    else:
                        # Create the same directory structure in the repository location
                        target_dir = os.path.join(location, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                else:
                    # Put all models directly in the repository location
                    target_dir = location
                
                target_path = os.path.join(target_dir, model_file.name)
                
                # Check if target already exists
                if os.path.exists(target_path):
                    if os.path.islink(target_path):
                        # Skip existing symlinks but count them
                        skipped_count += 1
                        continue
                    else:
                        # Skip real files
                        continue
                
                # Create the symlink with better error handling
                try:
                    # Double-check if the target exists right before creating the symlink
                    if os.path.exists(target_path):
                        # If it's a symlink, count it as skipped
                        if os.path.islink(target_path):
                            skipped_count += 1
                        # If it's a real file, we'll also skip it
                        continue
                    
                    # Create the symlink
                    os.symlink(model_file, target_path)
                    model_count += 1
                    
                    # Track model type for audit
                    model_type = ext.lstrip('.')
                    if model_type in model_types:
                        model_types[model_type] += 1
                    else:
                        model_types[model_type] = 1
                except FileExistsError:
                    # If the file exists despite our checks (race condition), count it as skipped
                    skipped_count += 1
                except Exception as e:
                    # Log other errors but continue processing
                    print(f"Error creating symlink for {model_file}: {str(e)}")
                    continue
        
        # Create audit trail
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_types_str = ", ".join([f"{count} {type}" for type, count in model_types.items()])
        audit_message = f"[{timestamp}] Synchronized {model_count} models ({model_types_str})"
        
        if skipped_count > 0:
            audit_message += f", skipped {skipped_count} existing symlinks"
        
        if sharded_count > 0 and not understands_shards:
            audit_message += f", ignored {sharded_count} sharded models"
        
        # Get the current tool repositories from config
        tool_repositories = config.config.get("tool_repositories", [])
        
        # Find and update the repository with the audit trail
        for repo in tool_repositories:
            if repo.get("name") == name:
                # Overwrite the audit trail with just the current message
                repo["audit_trail"] = [audit_message]
                break
        
        # Save the updated config
        config.config["tool_repositories"] = tool_repositories
        config.save_config()
        
        return jsonify({
            "success": True,
            "count": model_count,
            "skipped": skipped_count,
            "sharded": sharded_count,
            "audit": audit_message,
            "message": f"Successfully synchronized {model_count} models for {name}"
        })
    except Exception as e:
        # Log the error for server-side debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in sync_repository: {error_details}")
        
        # Return a more user-friendly error message
        error_message = str(e)
        if "File exists" in error_message:
            error_message = "Some files already exist in the target location. Please try again or manually remove conflicting files."
        
        return jsonify({
            "error": error_message,
            "details": str(e)
        }), 500

# API endpoint to scan for models in a directory
@app.route('/api/scan-models-in-dir', methods=['POST'])
def scan_models_in_directory():
    """Scan a directory for models and return a list of discovered models."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    search_startpoint = data.get('searchStartpoint')
    
    if not search_startpoint:
        return jsonify({"error": "Missing required field: searchStartpoint"}), 400
    
    # Check if directory exists
    if not os.path.isdir(search_startpoint):
        return jsonify({"error": f"Directory does not exist: {search_startpoint}"}), 400
    
    # We'll auto-detect the framework and always handle sharded models
    models, errors = scan_for_models(search_startpoint)
    
    if errors:
        # Return models with warnings
        return jsonify({
            "models": models,
            "warnings": errors
        })
    
    return jsonify({"models": models})

# API endpoint to add multiple models
@app.route('/api/add-models', methods=['POST'])
def add_multiple_models():
    """Add multiple models to the library."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    models = data.get('models', [])
    
    # Log the received models data
    print("Received models data:", json.dumps(models, indent=2))
    
    if not models:
        return jsonify({"error": "Missing required field: models"}), 400
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Check for existing models
    models_to_overwrite = []
    for model in models:
        model_path = model.get('path')
        model_name = model.get('name')
        
        if not model_path or not model_name:
            continue
        
        # Use provided publisher_name or extract from path
        publisher_name = model.get('publisher_name')
        if not publisher_name:
            publisher_name = os.path.basename(os.path.dirname(model_path))
        target_model_dir = os.path.join(model_library, publisher_name, model_name)
        
        if os.path.exists(target_model_dir):
            models_to_overwrite.append(f"{publisher_name}/{model_name}")
    
    if models_to_overwrite:
        return jsonify({
            "error": "Some models already exist in the library",
            "needsConfirmation": True,
            "modelsToOverwrite": models_to_overwrite
        }), 409
    
    # Process each model
    added_models = []
    errors = []
    framework = data.get('framework', 'unknown')
    
    for model in models:
        model_path = model.get('path')
        model_name = model.get('name')
        
        if not model_path or not model_name:
            errors.append(f"Missing path or name for model: {model}")
            continue
        
        # Use provided publisher_name or extract from path
        publisher_name = model.get("publisher_name")
        print(f"Publisher name from model data: {publisher_name}")
        if publisher_name is None or publisher_name == "":
            publisher_name = os.path.basename(os.path.dirname(model_path))
            print(f"Using extracted publisher name: {publisher_name}")
        print(f"Final publisher name: {publisher_name}")
        publisher_dir = os.path.join(model_library, publisher_name)
        
        # Create publisher directory if it doesn't exist
        if not os.path.exists(publisher_dir):
            os.makedirs(publisher_dir)
        
        # Create model directory path in the library
        target_model_dir = os.path.join(publisher_dir, model_name)
        
        try:
            # Create the model directory in the library
            print(f"Creating model directory: {target_model_dir}")
            os.makedirs(target_model_dir)
            
            # Find model files in the source directory
            model_extensions = config.get_model_extensions()
            model_files = []
            
            for ext in model_extensions:
                # Remove the leading dot if present to avoid double dots in the pattern
                ext_clean = ext[1:] if ext.startswith('.') else ext
                pattern = os.path.join(model_path, f"*.{ext_clean}")
                model_files.extend(glob.glob(pattern))
            
            if not model_files:
                # Clean up the empty directory we created
                os.rmdir(target_model_dir)
                errors.append(f"No model files found in {model_path}")
                continue
            
            # Always process sharded models
            # Group files by their base name (for sharded models)
            shard_groups = {}
            non_sharded_files = []
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if is_shard_file(filename):
                    base_name = extract_base_name(filename)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file_path)
                else:
                    non_sharded_files.append(file_path)
            
            # Copy each group of sharded files
            for base_name, shard_files in shard_groups.items():
                for shard_file in shard_files:
                    filename = os.path.basename(shard_file)
                    target_path = os.path.join(target_model_dir, filename)
                    print(f"Copying shard file: {shard_file} -> {target_path}")
                    shutil.copy2(shard_file, target_path)
            
            # Copy non-sharded files
            for file_path in non_sharded_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
            
            # Add the model to the database
            if shard_groups:
                # For sharded models, add each group
                for base_name, _ in shard_groups.items():
                    model_result, _ = add_model(
                        f"{model_name} ({base_name})", 
                        framework, 
                        target_model_dir,
                        {"is_sharded": True, "base_name": base_name}
                    )
                    added_models.append(model_result)
            else:
                # For non-sharded models
                # Add the model to the database
                model_result, _ = add_model(model_name, framework, target_model_dir)
                added_models.append(model_result)
                
        except Exception as e:
            errors.append(f"Error processing model {model_name}: {str(e)}")
            # Try to clean up if an error occurred
            if os.path.exists(target_model_dir):
                try:
                    shutil.rmtree(target_model_dir)
                except:
                    pass
    
    return jsonify({
        "message": f"Added {len(added_models)} models successfully",
        "added": added_models,
        "errors": errors
    })

# API endpoint to confirm overwrite of multiple models
@app.route('/api/confirm-overwrite-multiple', methods=['POST'])
def confirm_overwrite_multiple():
    """Confirm overwriting multiple existing models."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    models = data.get('models', [])
    framework = data.get('framework')
    handle_shards = data.get('handleShards', True)
    confirm = data.get('confirm', False)
    
    if not models or not framework:
        return jsonify({"error": "Missing required fields: models, framework"}), 400
    
    if not confirm:
        return jsonify({"error": "Confirmation is required"}), 400
    
    # Get the model library path
    model_library = config.get_model_library_path()
    
    # Process each model
    added_models = []
    errors = []
    framework = data.get('framework', 'unknown')
    
    for model in models:
        model_path = model.get('path')
        model_name = model.get('name')
        
        if not model_path or not model_name:
            errors.append(f"Missing path or name for model: {model}")
            continue
        
        # Use provided publisher_name or extract from path
        publisher_name = model.get("publisher_name")
        print(f"Publisher name from model data: {publisher_name}")
        if publisher_name is None or publisher_name == "":
            publisher_name = os.path.basename(os.path.dirname(model_path))
            print(f"Using extracted publisher name: {publisher_name}")
        print(f"Final publisher name: {publisher_name}")
        publisher_dir = os.path.join(model_library, publisher_name)
        
        # Create publisher directory if it doesn't exist
        if not os.path.exists(publisher_dir):
            os.makedirs(publisher_dir)
        
        # Create model directory path in the library
        target_model_dir = os.path.join(publisher_dir, model_name)
        
        try:
            # Remove existing model directory if it exists
            if os.path.exists(target_model_dir):
                # First, remove from database
                conn = sqlite3.connect(config.get_database_path())
                cursor = conn.cursor()
                cursor.execute("DELETE FROM models WHERE path LIKE ?", (f"{target_model_dir}%",))
                conn.commit()
                conn.close()
                
                # Then remove the directory
                shutil.rmtree(target_model_dir)
            
            # Create the model directory in the library
            print(f"Creating model directory: {target_model_dir}")
            os.makedirs(target_model_dir)
            
            # Find model files in the source directory
            model_extensions = config.get_model_extensions()
            model_files = []
            
            for ext in model_extensions:
                # Remove the leading dot if present to avoid double dots in the pattern
                ext_clean = ext[1:] if ext.startswith('.') else ext
                pattern = os.path.join(model_path, f"*.{ext_clean}")
                model_files.extend(glob.glob(pattern))
            
            if not model_files:
                # Clean up the empty directory we created
                os.rmdir(target_model_dir)
                errors.append(f"No model files found in {model_path}")
                continue
            
            # Always process sharded models
            # Group files by their base name (for sharded models)
            shard_groups = {}
            non_sharded_files = []
            
            for file_path in model_files:
                filename = os.path.basename(file_path)
                if is_shard_file(filename):
                    base_name = extract_base_name(filename)
                    if base_name not in shard_groups:
                        shard_groups[base_name] = []
                    shard_groups[base_name].append(file_path)
                else:
                    non_sharded_files.append(file_path)
            
            # Copy each group of sharded files
            for base_name, shard_files in shard_groups.items():
                for shard_file in shard_files:
                    filename = os.path.basename(shard_file)
                    target_path = os.path.join(target_model_dir, filename)
                    print(f"Copying shard file: {shard_file} -> {target_path}")
                    shutil.copy2(shard_file, target_path)
            
            # Copy non-sharded files
            for file_path in non_sharded_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_model_dir, filename)
                print(f"Copying file: {file_path} -> {target_path}")
                shutil.copy2(file_path, target_path)
            
            # Add the model to the database
            if shard_groups:
                # For sharded models, add each group
                for base_name, _ in shard_groups.items():
                    model_result, _ = add_model(
                        f"{model_name} ({base_name})", 
                        framework, 
                        target_model_dir,
                        {"is_sharded": True, "base_name": base_name}
                    )
                    added_models.append(model_result)
            else:
                # For non-sharded models
                # Add the model to the database
                model_result, _ = add_model(model_name, framework, target_model_dir)
                added_models.append(model_result)
                
        except Exception as e:
            errors.append(f"Error processing model {model_name}: {str(e)}")
            # Try to clean up if an error occurred
            if os.path.exists(target_model_dir):
                try:
                    shutil.rmtree(target_model_dir)
                except:
                    pass
    
    return jsonify({
        "message": f"Overwritten and added {len(added_models)} models successfully",
        "added": added_models,
        "errors": errors
    })





# API endpoint for server status
@app.route('/api/status', methods=['GET'])
def server_status():
    """Enhanced endpoint to check if the server is running with detailed status information."""
    import platform
    import psutil
    import os
    
    # Get process info
    process = psutil.Process(os.getpid())
    
    try:
        return jsonify({
            "status": "ok",
            "version": config.get_app_version(),
            "timestamp": datetime.datetime.now().isoformat(),
            "uptime": round((datetime.datetime.now() - datetime.datetime.fromtimestamp(process.create_time())).total_seconds()),
            "port": config.get_port(),
            "debug": config.get_debug_mode(),
            "platform": platform.system(),
            "python_version": platform.python_version()
        })
    except Exception as e:
        # Fallback to simple response if detailed info fails
        return jsonify({
            "status": "ok",
            "version": config.get_app_version(),
            "timestamp": datetime.datetime.now().isoformat()
        })


@app.route('/api/server/stop', methods=['POST'])
def stop_server():
    """Stop the server gracefully.
    
    This endpoint triggers a graceful shutdown of the current server.
    """
    import os
    import threading
    import time
    
    # Function to run the stop process after response is sent
    def run_stop():
        try:
            # Give time for the response to be sent
            time.sleep(1)
            # Log success and exit
            print("Shutting down server process")
            os._exit(0)  # Force immediate exit
        except Exception as e:
            print(f"Error during server shutdown: {str(e)}")
    
    try:
        # Start the stop process in a separate thread
        # This ensures the response is sent before the server exits
        stop_thread = threading.Thread(target=run_stop)
        stop_thread.daemon = True
        stop_thread.start()
        
        # Return success response
        return jsonify({
            "success": True, 
            "message": "Server shutdown initiated", 
            "details": "The server will shut down momentarily."
        })
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e), 
            "message": "Failed to stop server."
        })

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

