# LLM Model Manager Refactor Plan: Modern Architecture (MVC & Best Practices)

## Objective
Refactor the LLM Model Manager to a modern, maintainable, and scalable architecture using current best practices, including:
- **MVC (Model-View-Controller)** separation
- **Code reuse** via modularization and DRY principles
- **Testability** (unit/integration tests)
- **Extensibility** (easy to add new features)
- **Separation of concerns** (clear boundaries between data, logic, and presentation)
- **Consistent error handling & logging**
- **Configuration management** (12-factor principles)
- **Documentation & onboarding**

## Migration Strategy
- **Non-destructive**: Build new architecture in a parallel directory (`/llm_manager_v2/`), leaving the current app untouched until v2 is ready.
- **Incremental migration**: Move features one at a time, with integration tests to ensure parity.
- **Promotion**: Once v2 is complete and tested, switch over by updating entry points and deployment configs.

## Proposed Directory Structure

```
llm_manager_v2/
├── app.py                # Entry point (minimal, just launches the app)
├── config/               # Configuration schemas, env, and loaders
│   ├── __init__.py
│   └── settings.py
├── models/               # Data models and business logic
│   ├── __init__.py
│   ├── huggingface.py
│   ├── search_history.py
│   ├── user.py
│   └── ...
├── controllers/          # Route handlers and controllers
│   ├── __init__.py
│   ├── huggingface_controller.py
│   ├── settings_controller.py
│   └── ...
├── views/                # Templates and static assets
│   ├── templates/
│   │   ├── base.html
│   │   ├── huggingface.html
│   │   └── ...
│   └── static/
│       ├── css/
│       ├── js/
│       └── ...
├── services/             # External integrations, utility services
│   ├── __init__.py
│   ├── huggingface_service.py
│   └── ...
├── tests/                # Unit/integration tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_controllers.py
│   └── ...
├── migrations/           # DB migration scripts
├── requirements.txt
├── README.md
└── .env.example
```

## Key Best Practices to Adopt
- **MVC**: Strict separation between models (data), views (presentation), and controllers (logic/routing).
- **DRY & Code Reuse**: Shared utilities, base classes, and helpers for repeated logic.
- **Type Annotations**: Use type hints throughout for clarity and tooling.
- **Testing**: Pytest for all layers (models, controllers, services).
- **12-Factor Config**: All config from environment or config files, never hardcoded.
- **Blueprints/Modular Routing**: Flask Blueprints for feature modules.
- **Centralized Error Handling**: Custom error handlers, logging, and user feedback.
- **Documentation**: Docstrings, architecture docs, and onboarding guide.

## Step-by-Step Refactor Plan
1. **Bootstrap v2 Directory**: Scaffold `/llm_manager_v2/` with the above structure and minimal Flask app.
2. **Migrate Config System**: Move settings/config to `config/`, refactor to load from env and DB.
3. **Move Data Models**: Port models to `models/`, add type hints, and separate business logic.
4. **Add Controllers**: Create controllers for each major feature (Hugging Face, settings, etc.), using Blueprints.
5. **Template & Static Separation**: Move templates/static assets to `views/`, refactor for Jinja inheritance.
6. **Service Layer**: Abstract external APIs (Hugging Face, Ollama, etc.) into `services/`.
7. **Testing**: Add/port tests for each layer as it is migrated.
8. **Integration Testing**: Ensure v2 matches v1 in features and responses.
9. **Documentation**: Update README, add architecture and migration docs.
10. **Promotion**: Once v2 is stable and tested, switch entry points and deprecate v1.

## Ensuring a Smooth Transition
- **Parallel development**: No disruption to current users.
- **Automated tests**: Prevent regressions.
- **Clear migration docs**: For contributors and future maintainers.
- **Feature parity**: v2 must match v1 before promotion.

---

This plan ensures a clean, modern, maintainable codebase ready for future growth.
