# LLM Model Manager v2

This is the next-generation architecture for the LLM Model Manager, following modern best practices (MVC, modularization, testability, 12-factor config, etc.).

## Configuration System (v2)
- All app settings are loaded from environment variables, `.env` file, and (in future) the database.
- See `config/settings.py` for details. Use `get_config()` to access settings in code.
- Add new config keys as needed, following the type annotation pattern.

Development is parallel to v1 and will be promoted when feature parity and stability are achieved.

## Controllers & API Structure (v2)
- All API endpoints are organized as Flask Blueprints in `controllers/`.
- Major controllers:
  - `huggingface_controller.py`: Endpoints for Hugging Face search and history
  - `settings_controller.py`: Endpoints for config/settings
  - `models_controller.py`: Endpoints for model CRUD
- Register new Blueprints in `app.py` using `app.register_blueprint(...)`.
- Add new endpoints by extending the appropriate controller file.
