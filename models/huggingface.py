"""
Hugging Face integration for the LLM Model Manager.
Handles searching, downloading, and managing models from Hugging Face.
"""

import os
import shutil
import logging
from pathlib import Path
from huggingface_hub import HfApi, ModelFilter, login
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

from models import config
from models.database import add_model
from models.detection import detect_framework

# Configure logging
logger = logging.getLogger(__name__)

def initialize_huggingface():
    """Initialize the Hugging Face API with the stored token if available."""
    if not config.get_huggingface_enabled():
        logger.info("Hugging Face integration is disabled")
        return False
    
    token = config.get_huggingface_api_token()
    if not token:
        logger.warning("No Hugging Face API token configured")
        return False
    
    try:
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face API")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face API: {str(e)}")
        return False

def search_models(query, model_type=None, limit=50):
    """
    Search for models on Hugging Face.
    
    Args:
        query (str): Search query
        model_type (str, optional): Type of model to search for (e.g., 'llm', 'text-generation')
        limit (int, optional): Maximum number of results to return
        
    Returns:
        list: List of model information dictionaries
    """
    try:
        api = HfApi()
        filters = ModelFilter()
        
        if model_type:
            filters.task = model_type
        
        models = api.list_models(
            search=query,
            filter=filters,
            limit=limit,
            sort="downloads",
            direction=-1
        )
        
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "name": model.id.split('/')[-1],
                "author": model.id.split('/')[0] if '/' in model.id else "Unknown",
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "last_modified": model.last_modified.isoformat() if model.last_modified else None,
                "url": f"https://huggingface.co/{model.id}"
            })
        
        return results
    except Exception as e:
        logger.error(f"Error searching Hugging Face models: {str(e)}")
        return []

def download_model(model_id, filename=None):
    """
    Download a model from Hugging Face to the draft download area.
    
    Args:
        model_id (str): Hugging Face model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
        filename (str, optional): Specific filename to download, if None downloads all files
        
    Returns:
        dict: Status information about the download
    """
    if not initialize_huggingface():
        return {"success": False, "error": "Hugging Face API not initialized"}
    
    try:
        api = HfApi()
        draft_area = config.get_draft_download_area()
        os.makedirs(draft_area, exist_ok=True)
        
        # Get model files
        try:
            files = api.list_repo_files(model_id)
        except RepositoryNotFoundError:
            return {"success": False, "error": f"Model {model_id} not found"}
        
        # Filter files by extension if no specific filename is provided
        if not filename:
            model_extensions = config.get_model_extensions()
            model_files = [f for f in files if Path(f).suffix.lower() in model_extensions]
            if not model_files:
                return {
                    "success": False, 
                    "error": f"No model files found with supported extensions: {', '.join(model_extensions)}"
                }
        else:
            if filename not in files:
                return {"success": False, "error": f"File {filename} not found in model {model_id}"}
            model_files = [filename]
        
        # Download files
        downloaded_files = []
        for file in model_files:
            target_path = os.path.join(draft_area, os.path.basename(file))
            try:
                api.hf_hub_download(
                    repo_id=model_id,
                    filename=file,
                    local_dir=draft_area,
                    local_dir_use_symlinks=False
                )
                downloaded_files.append(target_path)
                logger.info(f"Downloaded {file} to {target_path}")
            except Exception as e:
                logger.error(f"Error downloading {file}: {str(e)}")
                return {"success": False, "error": f"Error downloading {file}: {str(e)}"}
        
        return {
            "success": True,
            "model_id": model_id,
            "files": downloaded_files,
            "location": draft_area
        }
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        return {"success": False, "error": str(e)}

def get_model_files(model_id):
    """Get a list of files for a specific model.
    
    Args:
        model_id (str): Hugging Face model ID
        
    Returns:
        dict: Dictionary with success status and list of files
    """
    if not initialize_huggingface():
        return {"success": False, "error": "Hugging Face API not initialized"}
    
    try:
        api = HfApi()
        
        # Get model files
        try:
            files = api.list_repo_files(model_id)
            return {"success": True, "files": files}
        except RepositoryNotFoundError:
            return {"success": False, "error": f"Model {model_id} not found"}
        except Exception as e:
            logger.error(f"Error listing files for model {model_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error getting model files: {str(e)}")
        return {"success": False, "error": str(e)}

def move_model_to_library(filename):
    """
    Move a model from the draft download area to the model library.
    
    Args:
        filename (str): Name of the file to move
        
    Returns:
        dict: Status information about the move operation
    """
    try:
        draft_area = config.get_draft_download_area()
        model_library = config.get_model_library_path()
        
        source_path = os.path.join(draft_area, filename)
        target_path = os.path.join(model_library, filename)
        
        if not os.path.exists(source_path):
            return {"success": False, "error": f"File {filename} not found in draft area"}
        
        # Ensure model library exists
        os.makedirs(model_library, exist_ok=True)
        
        # Move the file
        shutil.move(source_path, target_path)
        logger.info(f"Moved {filename} from draft area to model library")
        
        # Detect framework and add to database
        framework = detect_framework(target_path)
        add_model(target_path, framework)
        
        return {
            "success": True,
            "filename": filename,
            "new_location": target_path,
            "framework": framework
        }
    except Exception as e:
        logger.error(f"Error moving model to library: {str(e)}")
        return {"success": False, "error": str(e)}

def list_draft_models():
    """
    List all models in the draft download area.
    
    Returns:
        list: List of model file information
    """
    try:
        draft_area = config.get_draft_download_area()
        if not os.path.exists(draft_area):
            return []
        
        model_extensions = config.get_model_extensions()
        models = []
        
        for file in os.listdir(draft_area):
            file_path = os.path.join(draft_area, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in model_extensions:
                models.append({
                    "filename": file,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "modified": os.path.getmtime(file_path)
                })
        
        return models
    except Exception as e:
        logger.error(f"Error listing draft models: {str(e)}")
        return []

def delete_draft_model(filename):
    """
    Delete a model from the draft download area.
    
    Args:
        filename (str): Name of the file to delete
        
    Returns:
        dict: Status information about the delete operation
    """
    try:
        draft_area = config.get_draft_download_area()
        if not os.path.exists(draft_area):
            return {
                "success": False,
                "error": "Draft download area does not exist"
            }
        
        file_path = os.path.join(draft_area, filename)
        
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File {filename} not found in draft area"
            }
        
        if os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file {filename} from draft area")
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
            logger.info(f"Deleted directory {filename} from draft area")
        
        return {
            "success": True,
            "message": f"Successfully deleted {filename} from draft area"
        }
    except Exception as e:
        logger.error(f"Error deleting draft model {filename}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
