"""
Hugging Face module for the LLM Model Manager.

This module provides functions for:
- Searching for models on Hugging Face
- Getting trending models
- Downloading models
- Moving models from the draft area to the model library
- Managing the draft download area

The module implements a fallback mechanism with database caching when the API is not available or not configured.
"""

# Standard library imports
import os
import shutil
import logging
import traceback
import datetime
from pathlib import Path

# Third-party imports
from huggingface_hub import HfApi, ModelFilter, login
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import requests

# Application imports
from models import config
from models.database import add_model
from models.detection import detect_framework

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# AUTHENTICATION AND INITIALIZATION
# =============================================================================

def initialize_huggingface() -> bool:
    """Initialize the Hugging Face API with the stored token if available.
    
    This function attempts to authenticate with the Hugging Face API using
    the token stored in the application configuration. It will log appropriate
    messages based on the outcome of the authentication attempt.
    
    Returns:
        bool: True if authentication was successful, False otherwise
    """
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

# =============================================================================
# MODEL DISCOVERY AND SEARCH
# =============================================================================

def get_trending_models(limit: int = 50, model_type: str | None = None) -> list[dict]:
    """
    Get trending models from Hugging Face based on recent activity and popularity.
    This function uses the unified search_models approach with trending-specific parameters.
    
    This function attempts to retrieve popular models using the following strategy:
    1. First try to fetch from Hugging Face API using trending search parameters
    2. If API fails, fall back to cached popular models from search history
    
    Args:
        limit (int, optional): Maximum number of models to return. Defaults to 50.
        model_type (str, optional): Type of model to filter by. Defaults to None.
        
    Returns:
        list[dict]: List of trending model information dictionaries
    """
    try:
        # Try to get the user-configured limit from settings
        from models.config import get_setting
        configured_limit = get_setting("frameworks.huggingface.search_results_limit", default=50)
        # Use the smaller of the function parameter or configured limit
        limit = min(limit, int(configured_limit))
        logger.info(f"Using search results limit for trending models: {limit}")
        
        # Always add 'trending' to the search history with the persistent flag
        # This ensures there's always a trending tag available
        from models.search_history import add_search_query
        actual_model_type = model_type if model_type else "text-generation"
        add_search_query("trending", actual_model_type, is_persistent=True)
        
        # Get trending search configuration from settings
        trending_config = get_setting("frameworks.huggingface.trending_search", {
            "query": "",
            "model_type": "text-generation",
            "tags": ["gguf"],
            "sort_by": "last_modified",
            "sort_direction": -1
        })
        
        # If model_type is provided, override the config
        if model_type:
            trending_config["model_type"] = model_type
        
        # Use the search_models function with trending parameters
        # First try to get models from the API
        results = search_models(
            query=trending_config.get("query", ""),
            model_type=trending_config.get("model_type"),
            limit=limit,
            tags=trending_config.get("tags"),
            sort_by=trending_config.get("sort_by", "last_modified"),
            sort_direction=trending_config.get("sort_direction", -1),
            is_trending=True
        )
        
        # If we got results from the search, return them
        if results:
            logger.info(f"Successfully retrieved {len(results)} trending models from API")
            return results
        
        # If we get here, the API call failed or returned no results
        # Fall back to cached models from search history
        from models.search_history import get_popular_models
        cached_models = get_popular_models()
        
        if cached_models:
            logger.info(f"Using cached popular models from database (count: {len(cached_models)})")
            return cached_models[:limit]
        else:
            logger.warning("No cached models found and API unavailable or failed.")
            # Return an empty list if no models available
            return []
            
    except Exception as e:
        logger.error(f"Error fetching trending models: {str(e)}")
        logger.exception(e)
        
        # Attempt to get cached models as final fallback
        try:
            from models.search_history import get_popular_models as get_cached_popular_models
            cached_models = get_cached_popular_models()
            if cached_models:
                logger.info(f"Using cached popular models after error (count: {len(cached_models)})")
                return cached_models[:limit]
        except Exception as cache_error:
            logger.error(f"Error getting cached models: {str(cache_error)}")
        # If all else fails, return empty list
        return []


# Backwards compatibility alias
def get_popular_models(limit: int = 50, model_type: str | None = None) -> list[dict]:
    """
    Alias for get_trending_models for backward compatibility.
    
    Args:
        limit (int, optional): Maximum number of models to return. Defaults to 50.
        model_type (str, optional): Type of model to filter by. Defaults to None.
        
    Returns:
        list[dict]: List of trending model information dictionaries
    """
    return get_trending_models(limit, model_type)
        


def search_models(query: str = "", model_type: str | None = None, limit: int | None = None, 
              tags: list[str] | None = None, 
              sort_by: str = "downloads", 
              sort_direction: int = -1,
              is_trending: bool = False,
              no_fallback: bool = False) -> list[dict]:
    """
    Search for models on Hugging Face with enhanced parameters.
    This unified function handles both regular searches and trending model retrieval.
    
    Args:
        query (str, optional): Search query. Default is empty string.
        model_type (str, optional): Type of model to search for (e.g., 'text-generation')
        limit (int | None, optional): Maximum number of results to return, or None to use configured limit
        tags (list[str], optional): List of tags to filter by (e.g., ['gguf'])
        sort_by (str, optional): Field to sort results by (e.g., 'downloads', 'last_modified')
        sort_direction (int, optional): Sort direction (-1 for descending, 1 for ascending)
        is_trending (bool, optional): Whether this is a trending models search
        no_fallback (bool, optional): If True, never fall back to trending for failed searches
        
    Returns:
        list[dict]: List of model information dictionaries
    """
    try:
        # Check if Hugging Face API is enabled and configured
        from models.config import get_huggingface_enabled, get_huggingface_api_token, get_setting
        
        if not get_huggingface_enabled():
            logger.warning("Hugging Face API is not enabled in settings")
            # Return empty list as there's no fallback
            return []
            
        api_token = get_huggingface_api_token()
        if not api_token:
            logger.warning("Hugging Face API token is not configured")
            # Return empty list as there's no fallback
            return []
        
        # If limit is not specified, use the configured limit
        if limit is None:
            # Get the configured search results limit, default to 50 if not set
            limit = int(get_setting("frameworks.huggingface.search_results_limit", 50))
            logger.info(f"Using configured search results limit: {limit}")
        
        # Set up search parameters
        if tags is None:
            tags = ["gguf"]  # Default to GGUF models
            
        # Log the search parameters
        search_type = "trending models" if is_trending else "models"
        logger.info(f"Searching for {search_type} with query: '{query}', model_type: {model_type}, sort: {sort_by}")
        
        # Initialize API and filters
        api = HfApi(token=api_token)
        filters = ModelFilter()
        
        # Apply filters based on parameters
        if model_type:
            if hasattr(filters, 'task'):
                filters.task = model_type
            elif hasattr(filters, 'pipeline_tag'):
                filters.pipeline_tag = model_type
                
        # Apply tag filters
        if tags:
            filters.tags = tags
        
        # Perform the search
        logger.info(f"Querying Hugging Face API with filters: {filters}")
        
        # If this is a trending search, we may want different behavior
        if is_trending and sort_by == "last_modified":
            # For trending, get more results for better sorting
            models_generator = api.list_models(
                search=query,
                filter=filters,
                limit=limit*2,  # Get more results for trending to allow for better sorting
                sort=sort_by,
                direction=sort_direction
            )
            
            # Convert generator to list so we can work with it
            models = list(models_generator)
            
            # For trending, we'll do a hybrid sort that considers both recency and popularity
            if len(models) > limit:
                # Calculate a trending score that combines recency and downloads
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
        else:
            # Standard search
            models_generator = api.list_models(
                search=query,
                filter=filters,
                limit=limit,
                sort=sort_by,
                direction=sort_direction
            )
            # Convert generator to list
            models = list(models_generator)
        
        # Process results into a standardized format
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
                "url": f"https://huggingface.co/{model.id}",
                "is_trending": is_trending  # Mark if this was from a trending search
            })
        
        # Save the search to history if we got results
        if results:
            # Add the search to history with appropriate label
            search_label = "trending" if is_trending else query
            try:
                from models.search_history import add_search_query
                add_search_query(search_label, model_type)
                
                # If this is a trending search, also update popular models in database
                if is_trending:
                    from models.search_history import update_popular_models
                    update_popular_models(results)
            except Exception as e:
                logger.warning(f"Failed to update search history: {str(e)}")
        
        # If API returned no results for a non-trending search, we might want to return trending models instead
        # unless no_fallback is set to True
        if not results and not is_trending and not no_fallback:
            # Check if we have any trending models in the database already
            # If we do, don't fall back to trending (as requested)
            from models.search_history import get_popular_models
            cached_models = get_popular_models()
            
            if cached_models:
                logger.info(f"Search for '{query}' returned no results, but we have cached trending models")
                # We have cached trending models, so don't fall back to trending
                return []
                
            logger.info(f"Search for '{query}' returned no results, falling back to trending models")
            # Fall back to trending models only if we don't have any cached models
            return get_trending_models(limit, model_type)
            
        # If API returned no results, return an empty list
        if not results:
            search_type = "trending models" if is_trending else f"models matching '{query}'"
            logger.warning(f"Hugging Face API returned no {search_type}")
            return []
        
        return results
    except Exception as e:
        logger.error(f"Error searching Hugging Face models: {str(e)}")
        logger.exception(e)
        # Return empty list as there's no fallback
        return []


# =============================================================================
# MODEL DOWNLOAD AND MANAGEMENT
# =============================================================================

def download_model(model_id: str, filename: str | None = None) -> dict:
    """
    Download a model from Hugging Face to the draft download area.
    
    This function downloads a model from Hugging Face to the draft download area,
    following the library model format: publisher directory -> model directory -> files.
    If a specific filename is provided, only that file is downloaded. Otherwise,
    all files in the model repository are downloaded.
    
    Args:
        model_id (str): Hugging Face model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
        filename (str, optional): Specific filename to download. If None, downloads all files.
        
    Returns:
        dict: Status information about the download with the following keys:
            - success (bool): Whether the download was successful
            - message (str): A message describing the result
            - error (str, optional): Error message if download failed
            - downloaded_files (list, optional): List of downloaded files if successful
            - target_dir (str, optional): Directory where files were downloaded
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
        
        # Create publisher directory
        publisher_dir = os.path.join(draft_area, publisher)
        os.makedirs(publisher_dir, exist_ok=True)
        
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
        
        # Create a unique subfolder for each model file based on its name
        for file in model_files:
            # Get the base filename without extension
            base_filename = os.path.splitext(os.path.basename(file))[0]
            
            # Create a unique model subfolder
            # Use both model_name and base_filename to create a unique folder
            model_subfolder = f"{model_name}-{base_filename}"
            model_dir = os.path.join(publisher_dir, model_subfolder)
            os.makedirs(model_dir, exist_ok=True)
            
            # Set the target path for the file
            target_path = os.path.join(model_dir, os.path.basename(file))
            
            try:
                # Download the file to the unique subfolder
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
                    "model": model_subfolder,  # Use the unique subfolder name
                    "size": file_size
                })
                logger.info(f"Downloaded {file} to {target_path} ({file_size} bytes)")
            except Exception as e:
                logger.error(f"Error downloading {file}: {str(e)}")
                return {"success": False, "error": f"Error downloading {file}: {str(e)}"}
        
        # If we have downloaded files, use the location of the first file
        # as the model location for backward compatibility
        model_location = os.path.dirname(downloaded_files[0]["path"]) if downloaded_files else ""
        
        return {
            "success": True,
            "model_id": model_id,
            "publisher": publisher,
            "model_name": model_name,
            "files": downloaded_files,
            "location": model_location,
            "fileSize": file_size  # Include the file size in the response
        }
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        return {"success": False, "error": str(e)}

def get_model_files(model_id: str) -> dict:
    """Get a list of files for a specific model with their actual sizes.
    
    This function retrieves a list of files for a specific model from the Hugging Face API,
    including their actual sizes. It makes a direct API call to the Hugging Face API
    to get accurate file sizes rather than relying on estimated sizes.
    
    Args:
        model_id (str): Hugging Face model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
        
    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation was successful
            - files (list): List of file information dictionaries with the following keys:
                - filename (str): Name of the file
                - size (int): Size of the file in bytes
                - size_human (str): Human-readable size (e.g., '2.5 GB')
                - last_modified (str): Last modified date
                - gguf (bool): Whether the file is a GGUF file
                - safetensors (bool): Whether the file is a SafeTensors file
            - error (str, optional): Error message if operation failed
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

def move_model_to_library(filepath: str) -> dict:
    """
    Move a model from the draft download area to the model library.
    
    This function moves a model file from the draft download area to the model library,
    preserving the directory structure (publisher/model). It creates a unique directory
    for each model file to avoid conflicts and adds the model to the database.
    
    Args:
        filepath (str): Full path to the file to move
        
    Returns:
        dict: Status information about the move operation with the following keys:
            - success (bool): Whether the move was successful
            - message (str): A message describing the result
            - error (str, optional): Error message if move failed
            - source_path (str): Original path of the file
            - target_path (str): New path of the file in the model library
            - model_name (str): Name of the model as added to the database
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

# =============================================================================
# DRAFT AREA MANAGEMENT
# =============================================================================

def list_draft_models() -> dict:
    """
    List all models in the draft download area following the publisher/model structure.
    
    This function scans the draft download area and returns a list of all model files
    found, organized by publisher and model. It identifies GGUF and SafeTensors files
    and includes their sizes and last modified dates.
    
    Returns:
        dict: Dictionary with the following keys:
            - success (bool): Whether the operation was successful
            - models (list): List of model file information dictionaries with the following keys:
                - filepath (str): Full path to the file
                - filename (str): Name of the file
                - publisher (str): Publisher name
                - model (str): Model name
                - size (int): Size of the file in bytes
                - size_human (str): Human-readable size (e.g., '2.5 GB')
                - last_modified (str): Last modified date
                - gguf (bool): Whether the file is a GGUF file
                - safetensors (bool): Whether the file is a SafeTensors file
            - error (str, optional): Error message if operation failed
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

def delete_draft_model(filepath: str) -> dict:
    """
    Delete a model from the draft download area.
    
    This function deletes a model file or directory from the draft download area.
    If a file path is provided, it deletes the file and its parent directories if they are empty.
    If a directory path is provided, it deletes the entire directory structure.
    
    The function performs several safety checks:
    1. Verifies that the path exists
    2. Ensures the path is within the draft download area
    3. Checks if the path is a file or directory
    
    After deletion, it also cleans up empty parent directories.
    
    Args:
        filepath (str): Full path to the file or directory to delete
        
    Returns:
        dict: Status information about the delete operation with the following keys:
            - success (bool): Whether the deletion was successful
            - message (str): A message describing the result
            - error (str, optional): Error message if deletion failed
            - deleted_path (str): Path that was deleted
            - deleted_type (str): Type of the deleted item ('file' or 'directory')
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
