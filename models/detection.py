"""
Model detection functionality for the LLM Model Manager.
Handles framework detection, shard detection, and model scanning.
"""

import os
import re
import json
from pathlib import Path

import config
from models.database import add_model

def detect_framework(file_path):
    """Detect the framework based on file extension and name patterns."""
    path_str = str(file_path).lower()
    
    if path_str.endswith('.gguf') or path_str.endswith('.ggml'):
        return "llama.cpp"
    elif path_str.endswith('.bin') and ('pytorch' in path_str or 'torch' in path_str):
        return "transformers"
    elif path_str.endswith('.bin') or path_str.endswith('.safetensors'):
        return "transformers"
    elif path_str.endswith('.onnx'):
        return "onnx"
    elif path_str.endswith('.pt') or path_str.endswith('.pth'):
        return "pytorch"
    elif 'ollama' in path_str and (path_str.endswith('manifest') or '/manifests/' in path_str):
        return "ollama"
    else:
        return "unknown"

def is_shard_file(filename):
    """Check if a file is a shard based on naming pattern."""
    # Common patterns for sharded models
    patterns = [
        r'.*-\d{5}-of-\d{5}.*',  # Format: model-00001-of-00002
        r'.*-\d+-of-\d+.*',      # Format: model-1-of-2
        r'.*_\d+_of_\d+.*',      # Format: model_0001_of_0003
        r'.*\.\d+\.of\.\d+.*',  # Format: model.0001.of.0003
        r'.*\.\d{2}\..*',         # Format: model.00.safetensors
        r'.*\.\d+\..*',           # Format: model.0.safetensors
        r'.*_part-\d+.*',         # Format: model_part-0
        r'.*_part\d+.*',          # Format: model_part0
        r'.*-part-\d+.*',         # Format: model-part-0
        r'.*-part\d+.*',          # Format: model-part0
        r'.*\.part\.\d+\..*',     # Format: model.part.0.bin
        r'.*\.part\d+\..*',       # Format: model.part0.bin
        r'.*\.shard\.\d+\..*',    # Format: model.shard.0.safetensors
        r'.*\.shard\d+\..*',      # Format: model.shard0.safetensors
        r'.*-shard-\d+.*',        # Format: model-shard-0
        r'.*-shard\d+.*',         # Format: model-shard0
        r'.*_shard_\d+.*',        # Format: model_shard_0
        r'.*_shard\d+.*'          # Format: model_shard0
    ]
    
    for pattern in patterns:
        if re.match(pattern, filename):
            print(f"DEBUG: File {filename} identified as shard with pattern {pattern}")
            return True
    return False

def extract_base_name(filename):
    """Extract base name from a sharded file."""
    # Extract base name from different shard patterns
    base_name = filename
    print(f"DEBUG: Extracting base name from {filename}")
    
    # Pattern: model-00001-of-00002.bin
    match = re.match(r'(.*)-\d{5}-of-\d{5}(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model-1-of-2.bin
    match = re.match(r'(.*)-\d+-of-\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
        
    # Pattern: model_0001_of_0003.bin
    match = re.match(r'(.*)_\d+_of_\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
        
    # Pattern: model.0001.of.0003.bin
    match = re.match(r'(.*)\.\d+\.of\.\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model.00.safetensors or model.0.safetensors
    match = re.match(r'(.*)\.\d+(\.[^.]+)$', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model_part-0.bin
    match = re.match(r'(.*)_part-\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model_part0.bin
    match = re.match(r'(.*)_part\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model-part-0.bin
    match = re.match(r'(.*)-part-\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model-part0.bin
    match = re.match(r'(.*)-part\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model.part.0.bin
    match = re.match(r'(.*?)\.part\.\d+(\..*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model.part0.bin
    match = re.match(r'(.*?)\.part\d+(\..*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model.shard.0.safetensors
    match = re.match(r'(.*?)\.shard\.\d+(\..*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model.shard0.safetensors
    match = re.match(r'(.*?)\.shard\d+(\..*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model-shard-0.bin
    match = re.match(r'(.*)-shard-\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model-shard0.bin
    match = re.match(r'(.*)-shard\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model_shard_0.bin
    match = re.match(r'(.*)_shard_\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    # Pattern: model_shard0.bin
    match = re.match(r'(.*)_shard\d+(.*)', filename)
    if match:
        return match.group(1) + match.group(2)
    
    return base_name

def count_shards(base_name, file_list, extension):
    """Count how many shards exist for a base name."""
    print(f"DEBUG: Counting shards for base_name: {base_name}, extension: {extension}")
    print(f"DEBUG: Files to check: {file_list}")
    count = 0
    base_without_ext = os.path.splitext(base_name)[0]
    print(f"DEBUG: Base without extension: {base_without_ext}")
    
    for file in file_list:
        file_without_ext = os.path.splitext(file)[0]
        print(f"DEBUG: Checking file: {file}, without ext: {file_without_ext}")
        
        # Check for different shard patterns
        condition1 = base_without_ext in file_without_ext and file.endswith(extension) and is_shard_file(file)
        condition2 = base_name == extract_base_name(file) and file.endswith(extension)
        print(f"DEBUG: Condition 1: {condition1}, Condition 2: {condition2}")
        if condition1 or condition2:
            print(f"DEBUG: Found shard: {file}")
            count += 1
    
    return count

def scan_directory(directory_path):
    """Scan directory for model files and add them to the database."""
    try:
        print(f"DEBUG: Starting scan with directory_path = {directory_path}")
        print(f"DEBUG: MODEL_EXTENSIONS = {config.get_model_extensions()}")
        directory = Path(directory_path)
        print(f"DEBUG: Directory exists: {directory.exists()}, is_dir: {directory.is_dir()}")
        if not directory.exists() or not directory.is_dir():
            print(f"DEBUG: Directory does not exist or is not a directory: {directory_path}")
            return {"error": f"Directory does not exist: {directory_path}"}, 400
        
        models_added = 0
        processed_models = []
        existing_models = []
        processed_shards = set()  # Keep track of processed shard base names
        
        # Walk through the directory and find model files
        print(f"DEBUG: Starting scan of directory: {directory}")
        
        # List all top-level directories to verify we're scanning the right place
        print(f"DEBUG: Top-level directories in {directory}:")
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    print(f"DEBUG: Found directory: {item}")
        except Exception as e:
            print(f"DEBUG: Error listing directory contents: {e}")
            
        total_files_checked = 0
        model_extension_files = 0
        for root, dirs, files in os.walk(directory):
            print(f"DEBUG: Scanning directory: {root}")
            print(f"DEBUG: Found {len(dirs)} subdirectories and {len(files)} files")
            total_files_checked += len(files)
            
            # Count files with model extensions
            model_files = [f for f in files if Path(f).suffix.lower() in config.get_model_extensions()]
            model_extension_files += len(model_files)
            if model_files:
                print(f"DEBUG: Found {len(model_files)} potential model files in {root}: {model_files}")
            
            # Check for any .gguf files with 'of-' in the name
            shard_candidates = [f for f in files if f.endswith('.gguf') and 'of-' in f]
            if shard_candidates:
                print(f"DEBUG: Found potential shard files in {root}:")
                for candidate in shard_candidates:
                    print(f"DEBUG: Potential shard: {candidate}")
            for file in files:
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()
                print(f"DEBUG: Processing file: {file} with extension {file_ext}")
                
                # Skip non-model files
                if file_ext not in config.get_model_extensions():
                    print(f"DEBUG: Skipping {file} - extension {file_ext} not in {config.get_model_extensions()}")
                    continue
                
                # Check if this is a shard file
                print(f"DEBUG: Checking if {file} is a shard file")
                is_shard = is_shard_file(file)
                print(f"DEBUG: {file} is a shard file: {is_shard}")
                if is_shard:
                    base_name = extract_base_name(file)
                    base_path = str(Path(root) / base_name)
                    print(f"DEBUG: Base path for {file} is {base_path}")
                    print(f"DEBUG: Processed shards set: {processed_shards}")
                    print(f"DEBUG: Is base_path in processed_shards: {base_path in processed_shards}")
                    
                    # Skip if we've already processed this shard base
                    if base_path in processed_shards:
                        print(f"DEBUG: Skipping {file} as we've already processed {base_path}")
                        continue
                    
                    # Count shards for this base name
                    print(f"DEBUG: Counting shards for {base_name} in directory {root}")
                    shard_count = count_shards(base_name, os.listdir(root), file_ext)
                    print(f"DEBUG: Found {shard_count} shards for {base_name}")
                    
                    # Create a virtual path for the sharded model
                    virtual_path = base_path
                    print(f"DEBUG: Virtual path for sharded model: {virtual_path}")
                    
                    # Calculate total size of all shards
                    total_size_mb = 0
                    for shard_file in os.listdir(root):
                        shard_path = Path(root) / shard_file
                        if (base_name == extract_base_name(shard_file) and 
                            shard_file.endswith(file_ext) and 
                            shard_path.is_file()):
                            total_size_mb += shard_path.stat().st_size / (1024 * 1024)
                    
                    # Detect framework from the first shard
                    framework = detect_framework(file_path)
                    
                    # Create a friendly name from the base name
                    friendly_name = os.path.splitext(base_name)[0].split('/')[-1].replace('_', ' ').replace('-', ' ').title()
                    
                    # Add sharded model to database with special config
                    print(f"DEBUG: Creating sharded model with friendly name: {friendly_name}")
                    shard_config = {
                        "is_sharded": True,
                        "shard_count": shard_count,
                        "shard_pattern": file,
                        "shard_extension": file_ext
                    }
                    print(f"DEBUG: Sharded model config: {shard_config}")
                    
                    print(f"DEBUG: Adding sharded model to database: {friendly_name}")
                    result, status_code = add_model(
                        friendly_name, 
                        framework, 
                        virtual_path,
                        model_config=shard_config,
                        size_override=total_size_mb
                    )
                    print(f"DEBUG: Add sharded model result: {result}, status code: {status_code}")
                    
                    if status_code == 201:
                        models_added += 1
                        processed_models.append({
                            "name": friendly_name,
                            "framework": framework,
                            "path": virtual_path,
                            "size_mb": total_size_mb,
                            "is_sharded": True,
                            "shard_count": shard_count
                        })
                        print(f"DEBUG: Successfully added sharded model: {friendly_name}")
                    elif status_code == 400 and "already exists" in result.get("error", ""):
                        existing_models.append({
                            "name": friendly_name,
                            "framework": framework,
                            "path": virtual_path,
                            "size_mb": total_size_mb,
                            "is_sharded": True,
                            "shard_count": shard_count,
                            "already_exists": True
                        })
                        print(f"DEBUG: Model already exists: {friendly_name}")
                        
                    # Mark this shard base as processed
                    processed_shards.add(base_path)
                    print(f"DEBUG: Added {base_path} to processed_shards")
                    
                else:
                    # Regular non-sharded model file
                    framework = detect_framework(file_path)
                    
                    # Create a friendly name from the filename
                    friendly_name = os.path.splitext(file)[0].replace('_', ' ').replace('-', ' ').title()
                    
                    # Add model to database
                    result, status_code = add_model(
                        friendly_name, 
                        framework, 
                        str(file_path)
                    )
                    
                    if status_code == 201:
                        models_added += 1
                        processed_models.append({
                            "name": friendly_name,
                            "framework": framework,
                            "path": str(file_path),
                            "size_mb": file_path.stat().st_size / (1024 * 1024)
                        })
                    elif status_code == 400 and "already exists" in result.get("error", ""):
                        existing_models.append({
                            "name": friendly_name,
                            "framework": framework,
                            "path": str(file_path),
                            "size_mb": file_path.stat().st_size / (1024 * 1024),
                            "already_exists": True
                        })
        
        # Combine newly added and existing models
        all_models = processed_models + existing_models
        
        print(f"DEBUG: Scan complete. Added: {models_added}, Existing: {len(existing_models)}, Total models: {len(all_models)}")
        return {
            "added": models_added,
            "existing": len(existing_models),
            "models": all_models
        }, 200
        
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in scan_directory: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}, 500
