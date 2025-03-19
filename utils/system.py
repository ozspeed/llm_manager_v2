"""
System utilities for the LLM Model Manager.
Provides functions for system information and resource monitoring.
"""

import os
import psutil
import json

from models import config

def get_system_info():
    """Get system information including memory, CPU, and disk usage."""
    try:
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent
        }
        
        # CPU info
        cpu_info = {
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.1)
        }
        
        # Model library disk info
        library_disk = None
        model_library_path = config.get_model_library_path()
        if model_library_path and os.path.exists(model_library_path):
            try:
                disk_usage = psutil.disk_usage(model_library_path)
                library_disk = {
                    "path": model_library_path,
                    "total_gb": round(disk_usage.total / (1024**3), 2),
                    "used_gb": round(disk_usage.used / (1024**3), 2),
                    "free_gb": round(disk_usage.free / (1024**3), 2),
                    "percent": disk_usage.percent
                }
            except:
                # Fallback if we can't get disk usage for the model library
                library_disk = {
                    "path": model_library_path,
                    "error": "Could not get disk usage for this path"
                }
        
        # Database info
        db_size_mb = 0
        database_path = config.get_database_path()
        if os.path.exists(database_path):
            db_size_mb = round(os.path.getsize(database_path) / (1024 * 1024), 2)
        
        # Get model count from database
        model_count = 0
        try:
            import sqlite3
            conn = sqlite3.connect(database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM models")
            model_count = cursor.fetchone()[0]
            conn.close()
        except:
            pass
        
        return {
            "version": config.get_app_version(),
            "memory": memory_info,
            "cpu": cpu_info,
            "library_disk": library_disk,
            "database": {
                "path": database_path,
                "size_mb": db_size_mb,
                "model_count": model_count
            }
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500
