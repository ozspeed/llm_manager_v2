"""
Ollama-specific functionality for the LLM Model Manager.
Handles Ollama model detection, metadata extraction, and management.
"""

import subprocess
import re
import json

import config
from models.database import add_model

def parse_ollama_modelfile(modelfile_content):
    """Parse Ollama Modelfile content to extract metadata."""
    if not modelfile_content:
        return {}
    
    metadata = {
        "parameters": {},
        "system_prompt": None,
        "template": None,
        "license": None,
        "from_model": None,
        "display_name": None
    }
    
    # Parse FROM directive
    from_match = re.search(r'FROM\s+(.+?)(?:\s|$)', modelfile_content)
    if from_match:
        metadata["from_model"] = from_match.group(1).strip()
    
    # Parse SYSTEM directive
    system_match = re.search(r'SYSTEM\s+"""(.+?)"""', modelfile_content, re.DOTALL)
    if not system_match:
        system_match = re.search(r'SYSTEM\s+(.+?)(?:\n|$)', modelfile_content)
    if system_match:
        metadata["system_prompt"] = system_match.group(1).strip()
    
    # Parse TEMPLATE directive
    template_match = re.search(r'TEMPLATE\s+"""(.+?)"""', modelfile_content, re.DOTALL)
    if not template_match:
        template_match = re.search(r'TEMPLATE\s+(.+?)(?:\n|$)', modelfile_content)
    if template_match:
        metadata["template"] = template_match.group(1).strip()
    
    # Parse LICENSE directive
    license_match = re.search(r'LICENSE\s+"""(.+?)"""', modelfile_content, re.DOTALL)
    if not license_match:
        license_match = re.search(r'LICENSE\s+(.+?)(?:\n|$)', modelfile_content)
    if license_match:
        metadata["license"] = license_match.group(1).strip()
    
    # Parse PARAMETER directives
    parameter_matches = re.finditer(r'PARAMETER\s+(\w+)\s+(.+?)(?:\n|$)', modelfile_content)
    for match in parameter_matches:
        param_name = match.group(1).strip()
        param_value = match.group(2).strip()
        
        # Try to convert numeric values
        try:
            if '.' in param_value:
                param_value = float(param_value)
            else:
                param_value = int(param_value)
        except ValueError:
            # Keep as string if not numeric
            pass
        
        metadata["parameters"][param_name] = param_value
    
    return metadata

def scan_ollama_models(repositories=None):
    """Scan for Ollama models and add them to the database.
    
    Args:
        repositories: List of repositories to scan (e.g., ['ollama', 'local']). 
                     If None, only the default 'ollama' repository will be scanned.
    """
    try:
        print("\n==== Starting Ollama model scan ====")
        
        # Set default repositories if none provided
        if repositories is None or len(repositories) == 0:
            repositories = ["ollama"]
        
        print(f"Scanning repositories: {repositories}")
        
        # Check if Ollama is installed
        try:
            ollama_check = subprocess.run(
                ["which", "ollama"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if ollama_check.returncode != 0:
                print("Ollama not found in PATH")
                return {"models": [], "error": "Ollama not found in PATH"}
            
            print(f"Ollama check result: {ollama_check.stdout.strip()}")
        except Exception as e:
            print(f"Error checking for Ollama: {e}")
            return {"models": [], "error": f"Error checking for Ollama: {e}"}
        
        # Get list of Ollama models
        print("Running 'ollama list' command...")
        ollama_list = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if ollama_list.returncode != 0:
            print(f"Error running 'ollama list': {ollama_list.stderr}")
            return {"models": [], "error": f"Error running 'ollama list': {ollama_list.stderr}"}
        
        # Parse the output
        output_lines = ollama_list.stdout.strip().split('\n')
        print(f"Ollama list output:\n{ollama_list.stdout}")
        
        # Skip header line
        if len(output_lines) > 0 and "NAME" in output_lines[0]:
            output_lines = output_lines[1:]
        
        print(f"Found {len(output_lines)} lines in Ollama output")
        
        added_models = []
        for line in output_lines:
            if not line.strip():
                continue
                
            # Parse the line (format: NAME ID SIZE MODIFIED)
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                model_name = parts[0].strip()
                model_id = parts[1].strip()
                
                # Parse size (e.g., "42 GB")
                size_str = parts[2].strip()
                size_mb = 0
                try:
                    size_match = re.match(r'([\d.]+)\s*([KMG]B)', size_str)
                    if size_match:
                        size_value = float(size_match.group(1))
                        size_unit = size_match.group(2)
                        
                        if size_unit == 'KB':
                            size_mb = size_value / 1024
                        elif size_unit == 'MB':
                            size_mb = size_value
                        elif size_unit == 'GB':
                            size_mb = size_value * 1024
                except Exception as e:
                    print(f"Error parsing size '{size_str}': {e}")
                
                # Get modelfile for additional metadata
                try:
                    print(f"Getting modelfile for {model_name}...")
                    modelfile_result = subprocess.run(
                        ["ollama", "show", model_name, "--modelfile"], 
                        capture_output=True, 
                        text=True, 
                        check=False
                    )
                    
                    metadata = {}
                    if modelfile_result.returncode == 0:
                        modelfile_content = modelfile_result.stdout.strip()
                        print(f"Got modelfile for {model_name}, parsing...")
                        metadata = parse_ollama_modelfile(modelfile_content)
                        print(f"Parsed metadata: {metadata}")
                    else:
                        metadata = {}
                        print(f"Error getting modelfile for {model_name}: {modelfile_result.stderr}")
                except Exception as e:
                    metadata = {}
                    print(f"Error getting modelfile details: {e}")
                
                # Determine which repository this model belongs to
                repository = "ollama"  # Default repository
                for repo in repositories:
                    if model_name.startswith(f"{repo}/") or repo == "ollama":
                        repository = repo
                        break
                
                # Create a path for the Ollama model
                # This is a virtual path since Ollama manages its own storage
                ollama_path = f"ollama://{repository}/{model_name}"
                
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
                model_config = {
                    "ollama_id": model_id,
                    "modelfile": metadata,
                    "is_ollama": True,
                    "display_name": display_name,
                    "repository": repository
                }
                
                # Add model to database - use the display name instead of the raw model name
                print(f"Adding model to database: {display_name} (path: {ollama_path})")
                result, status_code = add_model(
                    display_name,  # Use the human-readable display name
                    "ollama", 
                    ollama_path,
                    model_config=model_config,
                    size_override=size_mb
                )
                print(f"Add model result: {result}, status code: {status_code}")
                
                # Add to results regardless of whether it was added successfully or already exists
                added_models.append({
                    "name": display_name,  # Use the human-readable display name
                    "original_name": model_name,  # Keep the original name for reference
                    "framework": "ollama",
                    "path": ollama_path,
                    "size_mb": size_mb,
                    "ollama_id": model_id,
                    "metadata": metadata,
                    "already_exists": status_code == 400 and "already exists" in str(result.get("error", ""))
                })
        
        print(f"Successfully processed {len(added_models)} Ollama models")
        return {"models": added_models}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Exception in scan_ollama_models: {e}")
        return {"models": [], "error": str(e)}
