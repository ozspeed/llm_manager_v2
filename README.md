# LLM Model Manager (MVP 1.0)

A web application for managing local LLM models across different frameworks. This tool helps you organize and track your local language models. Version 1.0 provides a stable foundation with core functionality for model management and detection.

## Features

- **Model Management**: Add, remove, and track local LLM models
- **Auto-Detection**: Automatically detect model frameworks based on file extensions
- **Library Scanning**: Scan directories to automatically add models
- **Resource Monitoring**: Track disk and memory usage
- **Framework Support**: Compatible with llama.cpp, Transformers, PyTorch, ONNX, and others
- **Sharded Model Support**: Improved detection and grouping of sharded model files
- **Database Management**: Reset database functionality for fresh starts
- **Error Handling**: Enhanced error handling and user feedback
- **Responsive UI**: Defensive programming to prevent UI issues

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open http://localhost:5000 in your browser

## Default Model Library

The application is configured to use the following path as the default model library:
```
/Volumes/Library_Bolt/AI Model Library
```

You can change this by modifying the `MODEL_LIBRARY_PATH` variable in `app.py` or by specifying a different path in the web interface.

## Supported Model Formats

The application automatically detects the following model formats:

- `.gguf`, `.ggml` - llama.cpp models
- `.bin` - Transformers models (if "pytorch" or "torch" in the filename)
- `.safetensors` - Transformers models
- `.onnx` - ONNX models
- `.pt`, `.pth` - PyTorch models

## Sharded Model Support

The application automatically detects and groups sharded model files. Sharded files are identified by the following patterns:

- `modelname-00001-of-00005.gguf` - Standard numbering with total count
- `modelname.00001.gguf` or `modelname-00001.gguf` - Simple numbered shards
- `modelname-part1.gguf` or `modelname.part1.gguf` - Part-based naming
- `modelname-shard1.gguf` or `modelname.shard1.gguf` - Shard-based naming

When sharded models are detected, they are displayed as a single model with the combined file size and a note indicating the number of shards.

## Recent Improvements in MVP 1.0

- **Enhanced Shard Detection**: Improved regex patterns for more reliable shard file detection
- **Robust Error Handling**: Better error handling throughout the application
- **Database Reset**: Added ability to reset the database from the UI
- **UI Stability**: Implemented defensive programming techniques to prevent blank screen issues
- **Performance Optimization**: Reduced redundant processing during directory scanning

## Roadmap for Future Versions

- Model inference testing
- Advanced filtering and search
- Backup/restore functionality
- User authentication
- Model metadata extraction
- Export/import functionality

## License

MIT
