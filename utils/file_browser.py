"""
File browser utility for the LLM Model Manager.
Provides functions for browsing directories for the directory selector.
"""

import os
import json
from flask import jsonify

def browse_directories(path):
    """
    Browse directories at the specified path.
    Returns a dictionary with current path, parent directory, and list of subdirectories.
    
    Args:
        path (str): The directory path to browse
        
    Returns:
        dict or tuple: Directory information as a dictionary on success, or
                     an error tuple (dict, status_code) on failure
    """
    try:
        # Normalize path
        path = os.path.normpath(path)
        
        # Get parent directory
        parent_dir = os.path.dirname(path) if path != '/' else None
        
        # List directories
        directories = []
        if os.path.exists(path) and os.path.isdir(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    directories.append({
                        "name": item,
                        "path": item_path
                    })
            
            # Sort directories by name
            directories.sort(key=lambda x: x["name"].lower())
        
        # Return raw data instead of a jsonify response
        return {
            "current_path": path,
            "parent_dir": parent_dir,
            "directories": directories
        }
    except Exception as e:
        # Return error tuple with raw dict and status code
        return {"error": str(e)}, 500
