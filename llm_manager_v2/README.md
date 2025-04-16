# LLM Model Manager v2

This is the next-generation architecture for the LLM Model Manager, following modern best practices (MVC, modularization, testability, 12-factor config, etc.).

## Configuration System (v2)
- All app settings are loaded from environment variables, `.env` file, and (in future) the database.
- See `config/settings.py` for details. Use `get_config()` to access settings in code.
- Add new config keys as needed, following the type annotation pattern.

Development is parallel to v1 and will be promoted when feature parity and stability are achieved.
