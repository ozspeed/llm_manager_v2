"""
Hugging Face Hub integration for the LLM Model Manager.
This module provides functions to search, browse, and download models from the Hugging Face Hub.
"""

import os
import json
import logging
from huggingface_hub import HfApi, hf_hub_download
import config

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('huggingface')

# Initialize Hugging Face API
hf_api = HfApi()

# Dictionary to track active downloads
active_downloads = {}

def filter_card_data(card_data):
    """
    Filter out extra_gated fields from the model card data.
    
    Args:
        card_data (dict): The model card data dictionary.
        
    Returns:
        dict: Filtered model card data.
    """
    if not card_data or not isinstance(card_data, dict):
        return card_data
    
    # Create a new dictionary with filtered keys
    filtered_data = {}
    for key, value in card_data.items():
        if not key.startswith('extra_gated'):
            filtered_data[key] = value
    
    return filtered_data


def move_model_to_library(model_id):
    """
    Move a model from the draft download area to the model library.
    
    Args:
        model_id (str): The Hugging Face model ID.
        
    Returns:
        dict: Result of the operation.
    """
    try:
        logger.info(f"Moving model {model_id} to library")
        
        # Get paths
        draft_area = config.get_draft_download_area()
        model_library = config.get_model_library_path()
        
        # Determine source and destination paths
        author = model_id.split('/')[0] if '/' in model_id else "huggingface"
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        source_dir = os.path.join(draft_area, author, model_name)
        dest_author_dir = os.path.join(model_library, author)
        dest_dir = os.path.join(dest_author_dir, model_name)
        
        # Check if source exists
        if not os.path.exists(source_dir):
            return {"error": f"Model {model_id} not found in draft area"}, 404
        
        # Create destination directories if needed
        if not os.path.exists(dest_author_dir):
            os.makedirs(dest_author_dir)
        
        # If destination already exists, handle it
        if os.path.exists(dest_dir):
            # Append timestamp to avoid conflicts
            import time
            timestamp = int(time.time())
            dest_dir = f"{dest_dir}_{timestamp}"
        
        # Move the directory
        import shutil
        shutil.move(source_dir, dest_dir)
        
        return {
            "success": True,
            "message": f"Model moved to {dest_dir}",
            "source": source_dir,
            "destination": dest_dir
        }, 200
    except Exception as e:
        logger.error(f"Error moving model to library: {str(e)}")
        return {"error": str(e)}, 500

def search_models(query=None, task=None, library=None, limit=50):
    """
    Search for models on Hugging Face Hub.
    
    Args:
        query (str, optional): Search query.
        task (str, optional): Task to filter by (e.g., 'text-generation', 'text-classification').
        library (str, optional): Library to filter by (e.g., 'pytorch', 'tensorflow').
        limit (int, optional): Maximum number of results to return.
        
    Returns:
        dict: Search results with model information.
    """
    try:
        logger.info(f"Searching Hugging Face Hub for models with query: {query}, task: {task}, library: {library}")
        
        # Set up filter parameters
        filter_params = []
        if task:
            filter_params.append(task)
        if library:
            filter_params.append(library)
            
        # Search for models
        models = hf_api.list_models(
            filter=filter_params if filter_params else None,
            search=query,
            limit=limit
        )
        
        # Format results
        results = []
        for model in models:
            result = {
                "id": model.id,
                "name": model.id.split('/')[-1] if '/' in model.id else model.id,
                "author": model.id.split('/')[0] if '/' in model.id else "Unknown",
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "last_modified": model.last_modified.isoformat() if model.last_modified else None,
                "library_name": model.library_name,
                "url": f"https://huggingface.co/{model.id}"
            }
            results.append(result)
        
        return {
            "models": results,
            "count": len(results)
        }, 200
    except Exception as e:
        logger.error(f"Error searching Hugging Face Hub: {str(e)}")
        return {"error": str(e)}, 500

def get_model_details(model_id):
    """
    Get detailed information about a specific model.
    
    Args:
        model_id (str): The Hugging Face model ID.
        
    Returns:
        dict: Model details.
    """
    try:
        logger.info(f"Getting details for model: {model_id}")
        
        # Get model info with file metadata (including sizes)
        model_info = hf_api.model_info(model_id, files_metadata=True)
        
        # Try to get README content
        readme_content = None
        try:
            from huggingface_hub import hf_hub_download
            import tempfile
            
            # Check if README.md exists in siblings
            readme_exists = any(sibling.rfilename == "README.md" for sibling in model_info.siblings)
            
            if readme_exists:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    readme_path = hf_hub_download(
                        repo_id=model_id,
                        filename="README.md",
                        repo_type="model",
                        local_dir=tmpdirname
                    )
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
        except Exception as e:
            logger.warning(f"Could not fetch README for {model_id}: {str(e)}")
        
        # Format results
        details = {
            "id": model_info.id,
            "name": model_info.id.split('/')[-1] if '/' in model_info.id else model_info.id,
            "author": model_info.id.split('/')[0] if '/' in model_info.id else "Unknown",
            "downloads": model_info.downloads,
            "likes": model_info.likes,
            "tags": model_info.tags,
            "pipeline_tag": model_info.pipeline_tag,
            "last_modified": model_info.last_modified.isoformat() if model_info.last_modified else None,
            "library_name": model_info.library_name,
            "url": f"https://huggingface.co/{model_info.id}",
            "description": filter_card_data(model_info.card_data.to_dict()) if hasattr(model_info, 'card_data') and model_info.card_data else "No description available",
            "readme": readme_content,
            "siblings": [
                {
                    "name": sibling.rfilename,
                    "size": sibling.size,
                    "type": os.path.splitext(sibling.rfilename)[1][1:] if '.' in sibling.rfilename else "unknown"
                }
                for sibling in model_info.siblings
            ]
        }
        
        return details, 200
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        return {"error": str(e)}, 500

def download_model(model_id, filename=None, revision=None, download_id=None):
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id (str): The Hugging Face model ID.
        filename (str, optional): Specific file to download. If None, downloads all model files.
        revision (str, optional): The revision to download (branch, tag, or commit).
        download_id (str, optional): Unique ID for tracking this download.
        
    Returns:
        dict: Download results.
    """
    import threading
    import uuid
    import time
    from huggingface_hub.utils import tqdm
    
    # Generate a download ID if not provided
    if not download_id:
        download_id = str(uuid.uuid4())
    
    # Create a download tracking object
    download_info = {
        "model_id": model_id,
        "status": "initializing",
        "progress": 0,
        "total_files": 0,
        "completed_files": 0,
        "current_file": "",
        "cancel_requested": False,
        "start_time": time.time(),
        "files": []
    }
    
    # Store in active downloads
    active_downloads[download_id] = download_info
    
    def download_thread():
        try:
            logger.info(f"Downloading model: {model_id}, file: {filename}, revision: {revision}, download_id: {download_id}")
            
            # Update status
            download_info["status"] = "preparing"
            
            # Get draft download area path
            draft_area = config.get_draft_download_area()
            
            # Create author directory if it doesn't exist
            author = model_id.split('/')[0] if '/' in model_id else "huggingface"
            author_dir = os.path.join(draft_area, author)
            if not os.path.exists(author_dir):
                os.makedirs(author_dir)
            
            # Create model directory
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            model_dir = os.path.join(author_dir, model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            downloaded_files = []
            
            # If filename is provided, download specific file
            if filename:
                download_info["total_files"] = 1
                download_info["current_file"] = filename
                
                # Check if cancel was requested
                if download_info["cancel_requested"]:
                    download_info["status"] = "cancelled"
                    return
                
                download_info["status"] = "downloading"
                
                file_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                
                download_info["completed_files"] = 1
                download_info["progress"] = 100
                downloaded_files = [file_path]
                download_info["files"].append({
                    "name": filename,
                    "path": file_path,
                    "status": "completed"
                })
            else:
                # Get model info to download all files
                model_info = hf_api.model_info(model_id)
                
                # Filter files by extension
                extensions = config.get_model_extensions()
                files_to_download = []
                
                for sibling in model_info.siblings:
                    file_ext = os.path.splitext(sibling.rfilename)[1]
                    if file_ext and (file_ext in extensions or file_ext.lower() in extensions):
                        files_to_download.append(sibling)
                
                download_info["total_files"] = len(files_to_download)
                
                # Download each file
                for i, sibling in enumerate(files_to_download):
                    # Check if cancel was requested
                    if download_info["cancel_requested"]:
                        download_info["status"] = "cancelled"
                        return
                    
                    download_info["status"] = "downloading"
                    download_info["current_file"] = sibling.rfilename
                    download_info["progress"] = int((i / len(files_to_download)) * 100)
                    
                    try:
                        file_path = hf_hub_download(
                            repo_id=model_id,
                            filename=sibling.rfilename,
                            revision=revision,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False
                        )
                        
                        downloaded_files.append(file_path)
                        download_info["completed_files"] += 1
                        download_info["files"].append({
                            "name": sibling.rfilename,
                            "path": file_path,
                            "status": "completed"
                        })
                    except Exception as e:
                        logger.error(f"Error downloading file {sibling.rfilename}: {str(e)}")
                        download_info["files"].append({
                            "name": sibling.rfilename,
                            "error": str(e),
                            "status": "failed"
                        })
        
            # Add model to database if download was successful and not cancelled
            if downloaded_files and not download_info["cancel_requested"]:
                download_info["status"] = "adding_to_database"
                
                from models.database import add_model
                from models.detection import is_shard_file, extract_base_name
                
                # Check if any of the downloaded files are shards
                sharded_files = [os.path.basename(f) for f in downloaded_files if is_shard_file(os.path.basename(f))]
                
                if sharded_files:
                    # Group sharded files by base name
                    shard_groups = {}
                    for filename in sharded_files:
                        base_name = extract_base_name(filename)
                        if base_name not in shard_groups:
                            shard_groups[base_name] = []
                        shard_groups[base_name].append(filename)
                    
                    # Add each shard group as a separate model
                    added_models = []
                    for base_name, _ in shard_groups.items():
                        model_result, _ = add_model(
                            f"{model_name} ({base_name})", 
                            model_dir, 
                            "huggingface", 
                            model_id
                        )
                        added_models.append(model_result)
                    
                    download_info["status"] = "completed"
                    return {
                        "success": True,
                        "message": f"Successfully downloaded {len(downloaded_files)} files and added {len(added_models)} models to the database",
                        "models": added_models,
                        "download_id": download_id
                    }, 200
                else:
                    # Add as a single model
                    model_result, _ = add_model(model_name, model_dir, "huggingface", model_id)
                    
                    download_info["status"] = "completed"
                    return {
                        "success": True,
                        "message": f"Successfully downloaded {len(downloaded_files)} files and added model to the database",
                        "model": model_result,
                        "download_id": download_id
                    }, 200
            elif download_info["cancel_requested"]:
                download_info["status"] = "cancelled"
                return {
                    "success": False,
                    "message": "Download was cancelled",
                    "download_id": download_id
                }, 200
            else:
                download_info["status"] = "failed"
                return {
                    "success": False,
                    "message": "No files were downloaded",
                    "download_id": download_id
                }, 500
        except Exception as e:
            logger.error(f"Error in download thread: {str(e)}")
            download_info["status"] = "failed"
            download_info["error"] = str(e)
            return {"error": str(e)}, 500
    
    # Start the download thread
    thread = threading.Thread(target=download_thread)
    thread.daemon = True
    thread.start()
    
    # Return immediately with the download ID
    return {
        "success": True,
        "message": "Download started",
        "download_id": download_id
    }, 200

def get_download_status(download_id):
    """
    Get the status of a download.
    
    Args:
        download_id (str): The download ID.
        
    Returns:
        dict: Download status.
    """
    if download_id not in active_downloads:
        return {"error": "Download not found"}, 404
    
    download_info = active_downloads[download_id]
    
    # Calculate elapsed time
    elapsed_time = time.time() - download_info["start_time"]
    
    # Calculate estimated time remaining
    eta = None
    if download_info["progress"] > 0:
        eta = (elapsed_time / download_info["progress"]) * (100 - download_info["progress"])
    
    status = {
        "download_id": download_id,
        "model_id": download_info["model_id"],
        "status": download_info["status"],
        "progress": download_info["progress"],
        "total_files": download_info["total_files"],
        "completed_files": download_info["completed_files"],
        "current_file": download_info["current_file"],
        "elapsed_time": elapsed_time,
        "eta": eta,
        "files": download_info["files"]
    }
    
    # Clean up completed or failed downloads after a while
    if download_info["status"] in ["completed", "failed", "cancelled"] and elapsed_time > 3600:  # 1 hour
        del active_downloads[download_id]
    
    return status, 200

def cancel_download(download_id):
    """
    Cancel a download.
    
    Args:
        download_id (str): The download ID.
        
    Returns:
        dict: Result of the cancellation.
    """
    if download_id not in active_downloads:
        return {"error": "Download not found"}, 404
    
    download_info = active_downloads[download_id]
    download_info["cancel_requested"] = True
    
    return {
        "success": True,
        "message": "Cancellation requested",
        "download_id": download_id
    }, 200

def get_model_versions(model_id):
    """
    Get available versions (tags, branches) for a specific model.
    
    Args:
        model_id (str): The Hugging Face model ID.
        
    Returns:
        dict: Available versions.
    """
    try:
        logger.info(f"Getting versions for model: {model_id}")
        
        # Get model info
        model_info = hf_api.model_info(model_id)
        
        # Get all available tags and branches
        tags = []
        branches = []
        
        try:
            # Get tags
            tags = hf_api.list_repo_refs(model_id).tags
            tags = [tag.name for tag in tags] if tags else []
            
            # Get branches
            branches = hf_api.list_repo_refs(model_id).branches
            branches = [branch.name for branch in branches] if branches else []
        except Exception as e:
            logger.warning(f"Error getting tags and branches: {str(e)}")
        
        # Format results
        versions = {
            "tags": tags,
            "branches": branches,
            "default_branch": "main"  # Default branch is usually 'main'
        }
        
        return versions, 200
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        return {"error": str(e)}, 500

def get_popular_models(limit=20):
    """
    Get a list of popular models from Hugging Face Hub.
    
    Args:
        limit (int, optional): Maximum number of results to return.
        
    Returns:
        dict: List of popular models.
    """
    try:
        logger.info(f"Getting popular models, limit: {limit}")
        
        # Get popular text generation models
        models = hf_api.list_models(
            filter="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit
        )
        
        # Format results
        results = []
        for model in models:
            result = {
                "id": model.id,
                "name": model.id.split('/')[-1] if '/' in model.id else model.id,
                "author": model.id.split('/')[0] if '/' in model.id else "Unknown",
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "last_modified": model.last_modified.isoformat() if model.last_modified else None,
                "library_name": model.library_name,
                "url": f"https://huggingface.co/{model.id}"
            }
            results.append(result)
        
        return {
            "models": results,
            "count": len(results)
        }, 200
    except Exception as e:
        logger.error(f"Error getting popular models: {str(e)}")
        return {"error": str(e)}, 500
