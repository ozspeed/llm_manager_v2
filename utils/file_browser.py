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
    Returns a list of subdirectories and parent directory.
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
        
        return jsonify({
            "current_path": path,
            "parent_dir": parent_dir,
            "directories": directories
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
