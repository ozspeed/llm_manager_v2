"""
Hugging Face Hub integration for the LLM Model Manager.
This module provides functions to search, browse, and download models from the Hugging Face Hub.
"""

import os
import json
import logging
from huggingface_hub import HfApi, ModelFilter, list_models, hf_hub_download
import config

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('huggingface')

# Initialize Hugging Face API
hf_api = HfApi()

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
        
        # Create model filter
        model_filter = ModelFilter()
        if task:
            model_filter.task = task
        if library:
            model_filter.library = library
            
        # Search for models
        models = list_models(
            filter=model_filter,
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
        
        # Get model info
        model_info = hf_api.model_info(model_id)
        
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
            "description": model_info.description,
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

def download_model(model_id, filename=None, revision=None):
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id (str): The Hugging Face model ID.
        filename (str, optional): Specific file to download. If None, downloads all model files.
        revision (str, optional): The revision to download (branch, tag, or commit).
        
    Returns:
        dict: Download results.
    """
    try:
        logger.info(f"Downloading model: {model_id}, file: {filename}")
        
        # Get model library path
        model_library = config.get_model_library_path()
        
        # Create author directory if it doesn't exist
        author = model_id.split('/')[0] if '/' in model_id else "huggingface"
        author_dir = os.path.join(model_library, author)
        if not os.path.exists(author_dir):
            os.makedirs(author_dir)
        
        # Create model directory
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        model_dir = os.path.join(author_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # If filename is provided, download specific file
        if filename:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            downloaded_files = [file_path]
        else:
            # Get model info to download all files
            model_info = hf_api.model_info(model_id)
            downloaded_files = []
            
            # Download each file
            for sibling in model_info.siblings:
                # Skip files that don't match our extensions
                extensions = config.get_model_extensions()
                file_ext = os.path.splitext(sibling.rfilename)[1]
                if file_ext not in extensions and file_ext.lower() not in extensions:
                    continue
                    
                file_path = hf_hub_download(
                    repo_id=model_id,
                    filename=sibling.rfilename,
                    revision=revision,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                downloaded_files.append(file_path)
        
        # Add model to database
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
                    "huggingface", 
                    model_dir,
                    {"is_sharded": True, "base_name": base_name}
                )
                added_models.append(model_result)
        else:
            # Add as a single model
            model_result, _ = add_model(model_name, "huggingface", model_dir)
            added_models = [model_result]
        
        return {
            "message": f"Successfully downloaded model {model_id}",
            "model_dir": model_dir,
            "downloaded_files": downloaded_files,
            "added_models": added_models
        }, 200
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
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
        models = list_models(
            filter=ModelFilter(task="text-generation"),
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
