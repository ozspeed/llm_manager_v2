#!/usr/bin/env python3

from models.detection import is_shard_file, extract_base_name, MODEL_EXTENSIONS

# Test files focusing on the 000X of 000Y format for .gguf files
test_files = [
    # The format mentioned by the user
    "model-0001-of-0003.gguf",
    "model-0002-of-0003.gguf",
    "model-0003-of-0003.gguf",
    
    # With different number of digits
    "model-00001-of-00002.gguf",
    "model-00002-of-00002.gguf",
    
    # With different separators
    "model_0001_of_0003.gguf",
    "model.0001.of.0003.gguf",
    
    # Original test cases
    "model-00001-of-00002.bin",
    "model-1-of-2.bin",
    "model.00.safetensors",
    "model.0.safetensors",
    "model_part-0.bin",
    "model_part0.bin",
    "model-part-0.bin",
    "model-part0.bin",
    "model.part.0.bin",
    "model.part0.bin",
    "model.shard.0.safetensors",
    "model.shard0.safetensors",
    "model-shard-0.bin",
    "model-shard0.bin",
    "model_shard_0.bin",
    "model_shard0.bin",
    
    # Add some non-shard files for comparison
    "model.gguf",
    "model.bin",
    "model.safetensors",
    "config.json"
]

print(f"MODEL_EXTENSIONS = {MODEL_EXTENSIONS}")
print("\nTesting shard detection:")
print("-" * 50)

for filename in test_files:
    is_shard = is_shard_file(filename)
    base_name = extract_base_name(filename) if is_shard else "N/A"
    print(f"File: {filename}")
    print(f"  Is shard: {is_shard}")
    print(f"  Base name: {base_name}")
    print("-" * 50)
