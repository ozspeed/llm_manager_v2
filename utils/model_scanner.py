"""Utility functions for scanning directories for models.

This module provides functions to scan directories for model files and identify models.
"""

import os
import glob
import uuid
import logging
from typing import List, Dict, Any, Tuple

import config
from models.detection import is_shard_file, extract_base_name

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def scan_for_models(directory: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Scan a directory and its subdirectories for model files.
    Auto-detects the framework and always handles sharded models.
    
    Args:
        directory: The directory to scan
        
    Returns:
        A tuple containing:
        - List of discovered models with metadata
        - List of errors encountered during scanning
    """
    models = []
    errors = []
    
    logger.debug(f"Scanning directory: {directory}")
    
    if not os.path.isdir(directory):
        error_msg = f"Directory does not exist: {directory}"
        logger.error(error_msg)
        errors.append(error_msg)
        return models, errors
    
    # Get model extensions from config
    model_extensions = config.get_model_extensions()
    logger.debug(f"Looking for files with extensions: {model_extensions}")
    
    # Scan the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        logger.debug(f"Checking directory: {root}")
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Check if this directory contains model files
        model_files = []
        for ext in model_extensions:
            # Remove the leading dot if present to avoid double dots in the pattern
            ext_clean = ext[1:] if ext.startswith('.') else ext
            pattern = os.path.join(root, f"*.{ext_clean}")
            found_files = glob.glob(pattern)
            if found_files:
                logger.debug(f"Found {len(found_files)} files with extension {ext} in {root}")
                logger.debug(f"Files: {found_files}")
            model_files.extend(found_files)
        
        if model_files:
            logger.debug(f"Directory {root} contains {len(model_files)} model files")
            # This directory contains model files, process it as a model
            try:
                # Auto-detect the framework based on the first file
                from models.detection import detect_framework
                detected_framework = "unknown"
                if model_files:
                    detected_framework = detect_framework(model_files[0])
                    logger.debug(f"Auto-detected framework: {detected_framework} for {root}")
                
                model_info = process_model_directory(root, model_files, detected_framework)
                if model_info:
                    logger.debug(f"Added model: {model_info['name']} from {root}")
                    models.append(model_info)
                else:
                    logger.warning(f"No model info returned for {root}")
            except Exception as e:
                error_msg = f"Error processing directory {root}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
    
    return models, errors

def process_model_directory(directory: str, model_files: List[str], framework: str) -> Dict[str, Any]:
    """
    Process a directory containing model files and extract metadata.
    
    Args:
        directory: The directory containing model files
        model_files: List of model files in the directory
        framework: The framework to use for model detection
        
    Returns:
        Dictionary containing model metadata
    """
    logger.debug(f"Processing directory: {directory} with {len(model_files)} files")
    
    # Extract model name from directory path
    model_name = os.path.basename(directory)
    logger.debug(f"Model name: {model_name}")
    
    # Group files by their base name (for sharded models)
    shard_groups = {}
    non_sharded_files = []
    
    for file_path in model_files:
        filename = os.path.basename(file_path)
        logger.debug(f"Checking if {filename} is a shard file")
        if is_shard_file(filename):
            base_name = extract_base_name(filename)
            logger.debug(f"File {filename} is a shard with base name: {base_name}")
            if base_name not in shard_groups:
                shard_groups[base_name] = []
            shard_groups[base_name].append(file_path)
        else:
            logger.debug(f"File {filename} is not a shard")
            non_sharded_files.append(file_path)
    
    logger.debug(f"Found {len(shard_groups)} shard groups and {len(non_sharded_files)} non-sharded files")
    if shard_groups:
        logger.debug(f"Shard groups: {list(shard_groups.keys())}")
    
    # Calculate total size
    total_size = 0
    for file_path in model_files:
        try:
            file_size = os.path.getsize(file_path)
            total_size += file_size
            logger.debug(f"File {os.path.basename(file_path)} size: {file_size / (1024 * 1024):.2f} MB")
        except (OSError, IOError) as e:
            logger.warning(f"Could not access file {file_path}: {str(e)}")
            pass  # Skip files that can't be accessed
    
    # Convert to MB
    size_mb = total_size / (1024 * 1024)
    logger.debug(f"Total size: {size_mb:.2f} MB")
    
    # Determine if this is a sharded model
    is_sharded = len(shard_groups) > 0
    shard_count = sum(len(files) for files in shard_groups.values())
    logger.debug(f"Is sharded: {is_sharded}, Shard count: {shard_count}")
    
    return {
        "id": str(uuid.uuid4()),  # Generate a temporary ID for the frontend
        "name": model_name,
        "path": directory,
        "framework": framework,
        "size_mb": size_mb,
        "is_sharded": is_sharded,
        "shard_count": shard_count,
        "file_count": len(model_files)
    }
