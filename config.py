"""
Configuration settings for the LLM Model Manager application.
"""

import os

# Application version
APP_VERSION = "1.0.0"  # MVP 1.0

# Default paths - can be overridden by environment variables
MODEL_LIBRARY_PATH = os.environ.get('MODEL_LIBRARY_PATH', "/Volumes/Library_Bolt/AI Model Library")
DATABASE_PATH = os.environ.get('DATABASE_PATH', "models.db")

# Supported model file extensions
MODEL_EXTENSIONS = ['.gguf', '.ggml', '.bin', '.safetensors', '.onnx', '.pt', '.pth']
