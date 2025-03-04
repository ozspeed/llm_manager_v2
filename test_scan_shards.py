#!/usr/bin/env python3

import os
from pathlib import Path
import re
from models.detection import is_shard_file, extract_base_name, count_shards, MODEL_EXTENSIONS

# Test directory with known sharded models
test_dir = "/Volumes/Library_Bolt/AI Model Library/Draft/Deepseek/Dynamic 2 bit Quant"

def scan_directory_for_shards(directory_path):
    """Scan directory for sharded model files."""
    print(f"Scanning directory: {directory_path}")
    
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"Directory does not exist: {directory_path}")
        return
    
    processed_shards = set()  # Keep track of processed shard base names
    
    # Get all files in the directory
    files = os.listdir(directory)
    print(f"Found {len(files)} files in directory")
    
    for file in files:
        file_path = directory / file
        file_ext = file_path.suffix.lower()
        
        print(f"Processing file: {file}")
        
        # Skip non-model files
        if file_ext not in MODEL_EXTENSIONS:
            print(f"Skipping {file} - extension {file_ext} not in {MODEL_EXTENSIONS}")
            continue
        
        # Check if this is a shard file
        is_shard = is_shard_file(file)
        print(f"Is shard file: {is_shard}")
        
        if is_shard:
            base_name = extract_base_name(file)
            print(f"Base name: {base_name}")
            
            base_path = str(directory / base_name)
            print(f"Base path: {base_path}")
            
            # Skip if we've already processed this shard base
            if base_path in processed_shards:
                print(f"Skipping {file} as we've already processed {base_path}")
                continue
            
            # Count shards for this base name
            shard_count = count_shards(base_name, files, file_ext)
            print(f"Found {shard_count} shards for {base_name}")
            
            # Mark this shard base as processed
            processed_shards.add(base_path)
            print(f"Added {base_path} to processed_shards")
            print("-" * 50)

if __name__ == "__main__":
    print(f"MODEL_EXTENSIONS = {MODEL_EXTENSIONS}")
    scan_directory_for_shards(test_dir)
    
    # Also test the specific DeepSeek model files we found earlier
    print("\nTesting specific DeepSeek model files:")
    test_files = [
        "DeepSeek-R1-UD-Q2_K_XL-00001-of-00005.gguf",
        "DeepSeek-R1-UD-Q2_K_XL-00002-of-00005.gguf",
        "DeepSeek-R1-UD-Q2_K_XL-00003-of-00005.gguf",
        "DeepSeek-R1-UD-Q2_K_XL-00004-of-00005.gguf",
        "DeepSeek-R1-UD-Q2_K_XL-00005-of-00005.gguf"
    ]
    
    print("-" * 50)
    for filename in test_files:
        is_shard = is_shard_file(filename)
        base_name = extract_base_name(filename) if is_shard else "N/A"
        print(f"File: {filename}")
        print(f"  Is shard: {is_shard}")
        print(f"  Base name: {base_name}")
        print("-" * 50)
