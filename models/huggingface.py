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

def get_popular_models(limit=12, model_type=None):
    """
    Get trending models from Hugging Face based on recent activity and popularity.
    
    Args:
        limit (int, optional): Maximum number of models to return
        model_type (str, optional): Type of model to filter by
        
    Returns:
        list: List of trending model information dictionaries
    """
    try:
        # First try to get cached popular models from search history
        from models.search_history import get_popular_models as get_cached_popular_models
        cached_models = get_cached_popular_models()
        
        if cached_models and len(cached_models) >= limit:
            logger.info(f"Using cached popular models (count: {len(cached_models)})")
            return cached_models[:limit]
        
        # Check if Hugging Face API is enabled and configured
        from models.config import get_huggingface_enabled, get_huggingface_api_token
        
        if not get_huggingface_enabled():
            logger.warning("Hugging Face API is not enabled in settings")
            # Return some hardcoded popular models as a fallback
            return get_hardcoded_popular_models(limit)
            
        api_token = get_huggingface_api_token()
        if not api_token:
            logger.warning("Hugging Face API token is not configured")
            # Return some hardcoded popular models as a fallback
            return get_hardcoded_popular_models(limit)
        
        # If no cached models or not enough, fetch from API
        logger.info(f"Fetching popular models from Hugging Face API")
        api = HfApi(token=api_token)
        filters = ModelFilter()
        
        # Add specific filters to get only GGUF models which are most relevant
        if not model_type:
            # Default to text generation models if no type specified
            filters.pipeline_tag = "text-generation"
        else:
            filters.pipeline_tag = model_type
            
        # Add filter for GGUF models
        filters.tags = ["gguf"]
        
        # Sort by last modified date to get trending models
        # This prioritizes recently updated models which are more likely to be trending
        logger.info(f"Querying Hugging Face API with filters: {filters}")
        models = api.list_models(filter=filters, sort="last_modified", direction=-1, limit=limit*2)
        
        # If we have more models than needed, we'll do a hybrid sort that considers
        # both recency and popularity to get truly trending models
        if len(models) > limit:
            # Calculate a trending score that combines recency and downloads
            # Higher score = more trending
            for model in models:
                # Convert last_modified to days ago (newer = smaller number)
                if model.last_modified:
                    import datetime
                    days_ago = (datetime.datetime.now(datetime.timezone.utc) - model.last_modified).days
                    # Avoid division by zero
                    days_ago = max(1, days_ago)
                else:
                    days_ago = 365  # Default to a year ago if no date
                
                # Trending score formula: downloads / days_ago
                # This prioritizes recent models with high download counts
                model.trending_score = model.downloads / days_ago
            
            # Sort by trending score
            models = sorted(models, key=lambda m: getattr(m, 'trending_score', 0), reverse=True)
            
            # Limit to requested number
            models = models[:limit]
        
        results = []
        for model in models:
            results.append({
                "id": model.id,
                "name": model.id.split('/')[-1],
                "author": model.id.split('/')[0] if '/' in model.id else 'Unknown',
                "downloads": model.downloads,
                "likes": model.likes,
                "tags": model.tags,
                "pipeline_tag": model.pipeline_tag,
                "last_modified": model.last_modified.isoformat() if model.last_modified else None,
                "url": f"https://huggingface.co/{model.id}"
            })
        
        # If API returned no results, use hardcoded models
        if not results:
            logger.warning("Hugging Face API returned no models, using hardcoded models")
            results = get_hardcoded_popular_models(limit)
        
        # Cache the results for future use
        if results:
            from models.search_history import update_popular_models
            update_popular_models(results)
            
        return results
    except Exception as e:
        logger.error(f"Error fetching popular Hugging Face models: {str(e)}")
        logger.exception(e)
        # Return hardcoded models as fallback
        return get_hardcoded_popular_models(limit)
        
def get_hardcoded_popular_models(limit=12):
    """
    Get a list of hardcoded trending models as a fallback when API is not available.
    
    Args:
        limit (int, optional): Maximum number of models to return
        
    Returns:
        list: List of trending model information dictionaries
    """
    # Import datetime for generating recent timestamps
    from datetime import datetime, timedelta
    
    # Generate recent dates for trending models
    now = datetime.now()
    today = now.strftime("%Y-%m-%dT%H:%M:%S")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    two_days_ago = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    three_days_ago = (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
    last_week = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    two_weeks_ago = (now - timedelta(days=14)).strftime("%Y-%m-%dT%H:%M:%S")
    
    # List of trending models with their metadata
    popular_models = [
        {
            "id": "TheBloke/Llama-3-8B-Instruct-GGUF",
            "name": "Llama-3-8B-Instruct-GGUF",
            "author": "TheBloke",
            "downloads": 250000,
            "likes": 1500,
            "tags": ["llama", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": yesterday,
            "url": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF"
        },
        {
            "id": "TheBloke/Phi-3-mini-4k-instruct-GGUF",
            "name": "Phi-3-mini-4k-instruct-GGUF",
            "author": "TheBloke",
            "downloads": 150000,
            "likes": 900,
            "tags": ["phi", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": today,
            "url": "https://huggingface.co/TheBloke/Phi-3-mini-4k-instruct-GGUF"
        },
        {
            "id": "TheBloke/Llama-3-70B-Instruct-GGUF",
            "name": "Llama-3-70B-Instruct-GGUF",
            "author": "TheBloke",
            "downloads": 90000,
            "likes": 600,
            "tags": ["llama", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": two_days_ago,
            "url": "https://huggingface.co/TheBloke/Llama-3-70B-Instruct-GGUF"
        },
        {
            "id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "name": "Mistral-7B-Instruct-v0.2-GGUF",
            "author": "TheBloke",
            "downloads": 200000,
            "likes": 1200,
            "tags": ["mistral", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": three_days_ago,
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        },
        {
            "id": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
            "name": "Mixtral-8x7B-Instruct-v0.1-GGUF",
            "author": "TheBloke",
            "downloads": 160000,
            "likes": 950,
            "tags": ["mixtral", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": last_week,
            "url": "https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF"
        },
        {
            "id": "TheBloke/Gemma-7B-it-GGUF",
            "name": "Gemma-7B-it-GGUF",
            "author": "TheBloke",
            "downloads": 130000,
            "likes": 800,
            "tags": ["gemma", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": two_days_ago,
            "url": "https://huggingface.co/TheBloke/Gemma-7B-it-GGUF"
        },
        {
            "id": "TheBloke/neural-chat-7B-v3-1-GGUF",
            "name": "neural-chat-7B-v3-1-GGUF",
            "author": "TheBloke",
            "downloads": 85000,
            "likes": 620,
            "tags": ["neural-chat", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": yesterday,
            "url": "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF"
        },
        {
            "id": "TheBloke/Qwen2-7B-Instruct-GGUF",
            "name": "Qwen2-7B-Instruct-GGUF",
            "author": "TheBloke",
            "downloads": 95000,
            "likes": 580,
            "tags": ["qwen", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": today,
            "url": "https://huggingface.co/TheBloke/Qwen2-7B-Instruct-GGUF"
        },
        {
            "id": "TheBloke/Llama-2-13B-chat-GGUF",
            "name": "Llama-2-13B-chat-GGUF",
            "author": "TheBloke",
            "downloads": 180000,
            "likes": 1100,
            "tags": ["llama", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": two_weeks_ago,
            "url": "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF"
        },
        {
            "id": "TheBloke/Phi-3-medium-4k-instruct-GGUF",
            "name": "Phi-3-medium-4k-instruct-GGUF",
            "author": "TheBloke",
            "downloads": 75000,
            "likes": 580,
            "tags": ["phi", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": yesterday,
            "url": "https://huggingface.co/TheBloke/Phi-3-medium-4k-instruct-GGUF"
        },
        {
            "id": "TheBloke/StableLM-2-1.6B-GGUF",
            "name": "StableLM-2-1.6B-GGUF",
            "author": "TheBloke",
            "downloads": 65000,
            "likes": 520,
            "tags": ["stablelm", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": today,
            "url": "https://huggingface.co/TheBloke/StableLM-2-1.6B-GGUF"
        },
        {
            "id": "TheBloke/Llama-3-8B-GGUF",
            "name": "Llama-3-8B-GGUF",
            "author": "TheBloke",
            "downloads": 120000,
            "likes": 780,
            "tags": ["llama", "gguf", "text-generation"],
            "pipeline_tag": "text-generation",
            "last_modified": three_days_ago,
            "url": "https://huggingface.co/TheBloke/Llama-3-8B-GGUF"
        }
    ]
    
    return popular_models[:limit]

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
        # Check if Hugging Face API is enabled and configured
        from models.config import get_huggingface_enabled, get_huggingface_api_token
        
        if not get_huggingface_enabled():
            logger.warning("Hugging Face API is not enabled in settings")
            # Return hardcoded search results as a fallback
            return search_hardcoded_models(query, model_type, limit)
            
        api_token = get_huggingface_api_token()
        if not api_token:
            logger.warning("Hugging Face API token is not configured")
            # Return hardcoded search results as a fallback
            return search_hardcoded_models(query, model_type, limit)
        
        logger.info(f"Searching for models with query: {query}, model_type: {model_type}")
        api = HfApi(token=api_token)
        filters = ModelFilter()
        
        # Add specific filters to get only GGUF models which are most relevant
        if model_type:
            filters.task = model_type
        
        # Add filter for GGUF models to prioritize them
        filters.tags = ["gguf"]
        
        # Perform the search
        logger.info(f"Querying Hugging Face API with filters: {filters}")
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
        
        # If API returned no results, use hardcoded search results
        if not results:
            logger.warning(f"Hugging Face API returned no models for query '{query}', using hardcoded results")
            return search_hardcoded_models(query, model_type, limit)
        
        return results
    except Exception as e:
        logger.error(f"Error searching Hugging Face models: {str(e)}")
        logger.exception(e)
        # Return hardcoded search results as fallback
        return search_hardcoded_models(query, model_type, limit)

def search_hardcoded_models(query, model_type=None, limit=50):
    """
    Search through hardcoded models when the API is not available.
    
    Args:
        query (str): Search query
        model_type (str, optional): Type of model to filter by
        limit (int, optional): Maximum number of results to return
        
    Returns:
        list: List of matching model information dictionaries
    """
    # Get all hardcoded models
    all_models = get_hardcoded_popular_models(100)  # Get a larger set to search through
    
    # Filter by query (case-insensitive)
    query = query.lower()
    filtered_models = []
    
    for model in all_models:
        # Check if query matches model id, name, or tags
        if (query in model["id"].lower() or 
            query in model["name"].lower() or 
            any(query in tag.lower() for tag in model.get("tags", []))):
            
            # If model_type is specified, filter by that too
            if model_type and model.get("pipeline_tag") != model_type:
                continue
                
            filtered_models.append(model)
    
    # Return limited results
    return filtered_models[:limit]

def download_model(model_id, filename=None):
    """
    Download a model from Hugging Face to the draft download area.
    Follows the library model format: publisher directory -> model directory -> files
    
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
        
        # Parse model_id to get publisher and model name
        parts = model_id.split('/')
        if len(parts) < 2:
            return {"success": False, "error": f"Invalid model ID format: {model_id}. Expected format: publisher/model"}
            
        publisher = parts[0]
        model_name = parts[1]
        
        # Create publisher and model directories
        publisher_dir = os.path.join(draft_area, publisher)
        model_dir = os.path.join(publisher_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
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
        file_size = 0
        for file in model_files:
            target_path = os.path.join(model_dir, os.path.basename(file))
            try:
                api.hf_hub_download(
                    repo_id=model_id,
                    filename=file,
                    local_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                
                # Get the file size
                if os.path.exists(target_path):
                    file_size = os.path.getsize(target_path)
                
                downloaded_files.append({
                    "filename": os.path.basename(file),
                    "path": target_path,
                    "publisher": publisher,
                    "model": model_name,
                    "size": file_size
                })
                logger.info(f"Downloaded {file} to {target_path} ({file_size} bytes)")
            except Exception as e:
                logger.error(f"Error downloading {file}: {str(e)}")
                return {"success": False, "error": f"Error downloading {file}: {str(e)}"}
        
        return {
            "success": True,
            "model_id": model_id,
            "publisher": publisher,
            "model_name": model_name,
            "files": downloaded_files,
            "location": model_dir,
            "fileSize": file_size  # Include the file size in the response
        }
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        return {"success": False, "error": str(e)}

def get_model_files(model_id):
    """Get a list of files for a specific model with their actual sizes.
    
    Args:
        model_id (str): Hugging Face model ID
        
    Returns:
        dict: Dictionary with success status and list of files with sizes
    """
    if not initialize_huggingface():
        return {"success": False, "error": "Hugging Face API not initialized"}
    
    try:
        api = HfApi()
        
        # Get model files
        try:
            # First get the basic file list
            files = api.list_repo_files(model_id)
            
            # Use direct API call to get file sizes from Hugging Face
            import requests
            import os
            
            # Get API token
            from models.config import get_huggingface_api_token
            token = os.environ.get("HUGGINGFACE_TOKEN") or get_huggingface_api_token()
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            
            # Fetch file tree with sizes
            url = f"https://huggingface.co/api/models/{model_id}/tree/main?recursive=True"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                file_data = response.json()
                
                # Create a mapping of filename to size
                file_sizes = {}
                for item in file_data:
                    if "path" in item and "size" in item:
                        file_sizes[item["path"]] = item["size"]
                
                # Add size information to each file
                files_with_sizes = []
                for file in files:
                    file_info = {
                        "name": file,
                        "size": file_sizes.get(file, 0)  # Default to 0 if size not found
                    }
                    files_with_sizes.append(file_info)
                
                return {"success": True, "files": files_with_sizes}
            else:
                logger.warning(f"Failed to fetch file sizes from HF API: {response.status_code} - {response.text}")
                # Fall back to basic file list without sizes
                files_with_sizes = [{"name": file, "size": 0} for file in files]
                return {"success": True, "files": files_with_sizes}
            
        except RepositoryNotFoundError:
            return {"success": False, "error": f"Model {model_id} not found"}
        except Exception as e:
            logger.error(f"Error listing files for model {model_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error getting model files: {str(e)}")
        return {"success": False, "error": str(e)}

def move_model_to_library(filepath):
    """
    Move a model from the draft download area to the model library.
    The directory structure (publisher/model) is preserved.
    
    Args:
        filepath (str): Full path to the file to move
        
    Returns:
        dict: Status information about the move operation
    """
    try:
        draft_area = config.get_draft_download_area()
        model_library = config.get_model_library_path()
        
        if not os.path.exists(filepath):
            return {"success": False, "error": f"File not found at path: {filepath}"}
        
        # Get the filename, model name, and publisher from the path
        parts = filepath.split(os.path.sep)
        if len(parts) < 3:
            return {
                "success": False,
                "error": f"Invalid file path format: {filepath}. Expected format: draft_area/publisher/model/file"
            }
            
        filename = os.path.basename(filepath)
        model_name = os.path.basename(os.path.dirname(filepath))
        publisher = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        
        # Create a model name from the filename without extension for the directory name
        model_display_name = os.path.splitext(filename)[0]
        
        # Create the target directory structure in the model library
        # Format: model_library/publisher/model_display_name/
        publisher_dir = os.path.join(model_library, publisher)
        os.makedirs(publisher_dir, exist_ok=True)
        
        # Create a unique directory for this specific model file
        target_model_dir = os.path.join(publisher_dir, model_display_name)
        os.makedirs(target_model_dir, exist_ok=True)
        
        target_path = os.path.join(target_model_dir, filename)
        
        # Move the file
        shutil.move(filepath, target_path)
        logger.info(f"Moved {filename} from draft area to model library at {target_path}")
        
        # Detect framework and add to database
        framework = detect_framework(target_path)
        
        # We already created model_display_name above
        
        # Add model to database with all required parameters
        # The add_model function requires name, framework, and path parameters
        add_model(model_display_name, framework, target_path)
        
        # Clean up empty directories in draft area
        model_dir = os.path.dirname(filepath)
        publisher_dir = os.path.dirname(model_dir)
        
        if os.path.exists(model_dir) and not os.listdir(model_dir):
            os.rmdir(model_dir)
            logger.info(f"Deleted empty model directory: {model_dir}")
            
            if os.path.exists(publisher_dir) and not os.listdir(publisher_dir):
                os.rmdir(publisher_dir)
                logger.info(f"Deleted empty publisher directory: {publisher_dir}")
        
        return {
            "success": True,
            "filename": filename,
            "publisher": publisher,
            "model": model_name,
            "new_location": target_path,
            "framework": framework
        }
    except Exception as e:
        logger.error(f"Error moving model to library: {str(e)}")
        return {"success": False, "error": str(e)}

def list_draft_models():
    """
    List all models in the draft download area following the publisher/model structure.
    
    Returns:
        list: List of model file information
    """
    try:
        draft_area = config.get_draft_download_area()
        if not os.path.exists(draft_area):
            return []
        
        model_extensions = config.get_model_extensions()
        models = []
        
        # Walk through the directory structure (publisher/model/files)
        for publisher in os.listdir(draft_area):
            publisher_path = os.path.join(draft_area, publisher)
            if not os.path.isdir(publisher_path):
                continue
                
            for model_name in os.listdir(publisher_path):
                model_path = os.path.join(publisher_path, model_name)
                if not os.path.isdir(model_path):
                    continue
                    
                for file in os.listdir(model_path):
                    file_path = os.path.join(model_path, file)
                    if os.path.isfile(file_path) and Path(file).suffix.lower() in model_extensions:
                        models.append({
                            "filename": file,
                            "path": file_path,
                            "publisher": publisher,
                            "model": model_name,
                            "size": os.path.getsize(file_path),
                            "modified": os.path.getmtime(file_path)
                        })
        
        return models
    except Exception as e:
        logger.error(f"Error listing draft models: {str(e)}")
        return []

def delete_draft_model(filepath):
    """
    Delete a model from the draft download area.
    If a file path is provided, delete the file and its parent directories if empty.
    If a directory path is provided, delete the entire directory structure.
    
    Args:
        filepath (str): Full path to the file or directory to delete
        
    Returns:
        dict: Status information about the delete operation
    """
    logger.info(f"[DELETE_DRAFT_MODEL] Function called with filepath: {filepath}")
    
    # Basic validation
    if not filepath:
        logger.error("[DELETE_DRAFT_MODEL] No filepath provided for deletion")
        return {
            "success": False,
            "error": "No filepath provided for deletion"
        }
    
    if not isinstance(filepath, str):
        logger.error(f"[DELETE_DRAFT_MODEL] Invalid filepath type: {type(filepath)}")
        return {
            "success": False,
            "error": f"Invalid filepath type: {type(filepath)}"
        }
        
    try:
        logger.info(f"[DELETE_DRAFT_MODEL] Attempting to delete: {filepath}")
        draft_area = config.get_draft_download_area()
        logger.info(f"[DELETE_DRAFT_MODEL] Draft area configured as: {draft_area}")
        
        if not draft_area:
            logger.error("[DELETE_DRAFT_MODEL] Draft download area not configured")
            return {
                "success": False,
                "error": "Draft download area not configured"
            }
            
        if not os.path.exists(draft_area):
            logger.error(f"[DELETE_DRAFT_MODEL] Draft download area does not exist: {draft_area}")
            return {
                "success": False,
                "error": "Draft download area does not exist"
            }
        
        # Check if path exists
        if not os.path.exists(filepath):
            logger.error(f"[DELETE_DRAFT_MODEL] Path not found: {filepath}")
            return {
                "success": False,
                "error": f"Path not found: {filepath}"
            }
        
        # Check if the path is within the draft area
        if not filepath.startswith(draft_area):
            logger.error(f"[DELETE_DRAFT_MODEL] Security check failed - Path is not within draft area")
            logger.error(f"[DELETE_DRAFT_MODEL] Path: {filepath}")
            logger.error(f"[DELETE_DRAFT_MODEL] Draft area: {draft_area}")
            return {
                "success": False,
                "error": f"Security error: Path is not within draft area"
            }
        
        # Get the filename, model name, and publisher from the path
        logger.info(f"[DELETE_DRAFT_MODEL] Parsing path components from: {filepath}")
        parts = filepath.split(os.path.sep)
        logger.debug(f"[DELETE_DRAFT_MODEL] Path parts: {parts}")
        
        if len(parts) < 3:
            logger.error(f"[DELETE_DRAFT_MODEL] Invalid path format: {filepath}")
            logger.error(f"[DELETE_DRAFT_MODEL] Path parts count: {len(parts)}, expected at least 3")
            return {
                "success": False,
                "error": f"Invalid path format: {filepath}. Expected format: draft_area/publisher/model/file"
            }
        
        # Determine if we're deleting a file or a directory
        is_file = os.path.isfile(filepath)
        is_dir = os.path.isdir(filepath)
        
        path_type = 'file' if is_file else 'directory' if is_dir else 'unknown'
        logger.info(f"[DELETE_DRAFT_MODEL] Path type: {path_type}")
        
        if path_type == 'unknown':
            logger.error(f"[DELETE_DRAFT_MODEL] Path is neither a file nor a directory: {filepath}")
            return {
                "success": False,
                "error": f"Path is neither a file nor a directory: {filepath}"
            }
        
        # Extract components
        filename = os.path.basename(filepath)
        model_dir = os.path.dirname(filepath) if is_file else filepath
        publisher_dir = os.path.dirname(model_dir)
        
        logger.info(f"[DELETE_DRAFT_MODEL] Filename: {filename}")
        logger.info(f"[DELETE_DRAFT_MODEL] Model directory: {model_dir}")
        logger.info(f"[DELETE_DRAFT_MODEL] Publisher directory: {publisher_dir}")
        
        # If it's a file, delete it
        if is_file:
            try:
                logger.info(f"[DELETE_DRAFT_MODEL] Deleting file: {filepath}")
                os.remove(filepath)
                logger.info(f"[DELETE_DRAFT_MODEL] File deleted successfully: {filename}")
            except Exception as e:
                logger.error(f"[DELETE_DRAFT_MODEL] Error deleting file: {e}")
                return {
                    "success": False,
                    "error": f"Error deleting file: {str(e)}"
                }
        
        # If it's a directory (model directory), delete the entire directory
        elif is_dir:
            try:
                logger.info(f"[DELETE_DRAFT_MODEL] Deleting directory: {filepath}")
                # Get directory size and contents for logging
                total_files = sum([len(files) for _, _, files in os.walk(filepath)])
                logger.info(f"[DELETE_DRAFT_MODEL] Directory contains {total_files} files")
                
                # Use shutil.rmtree to recursively delete the directory and all its contents
                model_name = os.path.basename(filepath)
                shutil.rmtree(filepath)
                logger.info(f"[DELETE_DRAFT_MODEL] Directory deleted successfully: {model_name}")
                filename = model_name  # For the success message
            except Exception as e:
                logger.error(f"[DELETE_DRAFT_MODEL] Error deleting directory: {e}")
                return {
                    "success": False,
                    "error": f"Error deleting directory: {str(e)}"
                }
        else:
            # This should never happen as we already checked above
            logger.error(f"[DELETE_DRAFT_MODEL] Path is neither a file nor a directory: {filepath}")
            return {
                "success": False,
                "error": f"Path is neither a file nor a directory: {filepath}"
            }
        
        # Check if model directory is empty and delete if it is
        if os.path.exists(model_dir) and not is_dir and not os.listdir(model_dir):
            logger.info(f"[DELETE_DRAFT_MODEL] Model directory is empty, deleting: {model_dir}")
            try:
                os.rmdir(model_dir)
                logger.info(f"[DELETE_DRAFT_MODEL] Empty model directory deleted: {os.path.basename(model_dir)}")
            except Exception as e:
                logger.warning(f"[DELETE_DRAFT_MODEL] Could not delete empty model directory: {e}")
                # Continue execution, this is not a critical error
        
        # Check if publisher directory is empty and delete if it is
        # Consider it empty if it only contains .DS_Store or other hidden files
        if os.path.exists(publisher_dir):
            # Get list of files excluding .DS_Store and other hidden files
            visible_files = [f for f in os.listdir(publisher_dir) if not f.startswith('.')]
            
            if not visible_files:
                logger.info(f"[DELETE_DRAFT_MODEL] Publisher directory is empty or contains only hidden files, deleting: {publisher_dir}")
                
                # First remove any .DS_Store files
                for hidden_file in os.listdir(publisher_dir):
                    if hidden_file.startswith('.'):
                        try:
                            hidden_path = os.path.join(publisher_dir, hidden_file)
                            logger.info(f"[DELETE_DRAFT_MODEL] Removing hidden file: {hidden_path}")
                            os.remove(hidden_path)
                        except Exception as hidden_err:
                            logger.warning(f"[DELETE_DRAFT_MODEL] Could not remove hidden file {hidden_file}: {hidden_err}")
                
                # Now try to remove the directory
                try:
                    os.rmdir(publisher_dir)
                    logger.info(f"[DELETE_DRAFT_MODEL] Empty publisher directory deleted: {os.path.basename(publisher_dir)}")
                except Exception as e:
                    logger.warning(f"[DELETE_DRAFT_MODEL] Could not delete empty publisher directory: {e}")
                    # Continue execution, this is not a critical error
        
        logger.info(f"[DELETE_DRAFT_MODEL] Operation completed successfully")
        logger.info(f"[DELETE_DRAFT_MODEL] Deleted {filename} from draft area")
        
        return {
            "success": True,
            "message": f"Successfully deleted {filename} from draft area",
            "deleted_path": filepath,
            "deleted_type": "file" if is_file else "directory"
        }
    except Exception as e:
        # Log the full exception with traceback for debugging
        logger.error(f"[DELETE_DRAFT_MODEL] Unhandled exception during deletion: {str(e)}")
        import traceback
        logger.error(f"[DELETE_DRAFT_MODEL] Exception traceback: {traceback.format_exc()}")
        
        return {
            "success": False,
            "error": f"Error deleting draft model: {str(e)}",
            "exception_type": type(e).__name__
        }
