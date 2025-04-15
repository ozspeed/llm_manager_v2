# Changelog

All notable changes to the LLM Model Manager will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-04-14

### Added
- Comprehensive Hugging Face integration
  - Search for models with filtering options
  - Browse popular and trending models
  - Download models to a draft area
  - Move models to library with proper organization
  - Accurate file size reporting from the Hugging Face API
- Draft Download Area for staging downloaded models
- New API endpoints for fetching popular models and recent searches
- Improved error handling for API responses

### Changed
- Enhanced code organization and documentation
- Improved module structure with clear separation of concerns
- Better docstrings and section comments throughout the codebase
- Updated README with comprehensive documentation

### Fixed
- Fixed "add_model() missing 1 required positional argument: 'path'" error
- Resolved "Error loading draft models: null is not an object" error
- Added comprehensive null checking for DOM elements in JavaScript code
- Fixed display of file sizes in the model info card and download options dialog
- Improved error handling and cleanup of empty directories after moving models

## [1.0.1] - 2025-03-04

### Added
- Enhanced shard detection with improved regex patterns
- Comprehensive debug logging for better troubleshooting
- Test scripts for shard detection and directory access

### Changed
- Refactored codebase into a modular structure
- Fixed and enhanced the system resources display in the UI

## [1.0.0] - 2025-02-15

### Added
- Initial release with basic model management functionality
- Framework detection for various model types
- Simple UI for model management
- Support for llama.cpp, Transformers, PyTorch, ONNX, and Ollama models
- Basic sharded model support
- Database management functionality
