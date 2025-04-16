#!/usr/bin/env python3

import os
from pathlib import Path

# Test directory with known sharded models
test_dir = "/Volumes/Library_Bolt/AI Model Library/Draft/Deepseek/Dynamic 2 bit Quant"

def test_directory_access(directory_path):
    """Test if we can access the directory and list its contents."""
    print(f"Testing directory access: {directory_path}")
    
    directory = Path(directory_path)
    print(f"Directory exists: {directory.exists()}")
    print(f"Is directory: {directory.is_dir()}")
    
    if directory.exists() and directory.is_dir():
        try:
            print(f"Directory contents:")
            for item in directory.iterdir():
                print(f"  {item.name} - {'Directory' if item.is_dir() else 'File'}")
                
            # Look for .gguf files with 'of-' in the name
            gguf_files = [f for f in directory.iterdir() if f.is_file() and f.name.endswith('.gguf') and 'of-' in f.name]
            print(f"\nFound {len(gguf_files)} .gguf files with 'of-' in the name:")
            for f in gguf_files:
                print(f"  {f.name}")
                
        except Exception as e:
            print(f"Error accessing directory: {e}")
    else:
        print(f"Cannot access directory: {directory_path}")
        
    # Try using os.walk
    print("\nTrying os.walk:")
    try:
        for root, dirs, files in os.walk(directory):
            print(f"Walking directory: {root}")
            print(f"Found {len(dirs)} subdirectories and {len(files)} files")
            
            # Check for any .gguf files with 'of-' in the name
            shard_candidates = [f for f in files if f.endswith('.gguf') and 'of-' in f]
            if shard_candidates:
                print(f"Found potential shard files:")
                for candidate in shard_candidates:
                    print(f"  {candidate}")
            break  # Just check the top level
    except Exception as e:
        print(f"Error with os.walk: {e}")

if __name__ == "__main__":
    test_directory_access(test_dir)
