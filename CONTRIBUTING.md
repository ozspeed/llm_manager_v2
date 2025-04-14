# Contributing to LLM Model Manager

Thank you for your interest in contributing to the LLM Model Manager! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. Steps to reproduce the bug
3. Expected behavior
4. Actual behavior
5. Screenshots if applicable
6. Your environment (OS, browser, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue with:

1. A clear, descriptive title
2. A detailed description of the proposed feature
3. Any relevant mockups or examples
4. Why this feature would be beneficial to the project

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests if available
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

## Development Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Coding Standards

### Python

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions small and focused on a single task
- Use type hints where appropriate

### JavaScript

- Use ES6+ features where possible
- Follow consistent indentation (2 spaces)
- Use meaningful variable and function names
- Add comments for complex logic
- Use async/await for asynchronous operations

### HTML/CSS

- Use semantic HTML elements
- Follow a consistent naming convention for CSS classes
- Keep CSS organized and modular
- Ensure responsive design principles are followed

## Testing

- Add tests for new features when possible
- Ensure existing tests pass before submitting a pull request
- Test your changes in different browsers and environments

## Documentation

- Update the README.md if your changes affect the user experience
- Add or update docstrings for new or modified functions
- Update the CHANGELOG.md for significant changes

## Branching Strategy

- `main`: Stable production code
- `develop`: Development branch for upcoming releases
- Feature branches: Named as `feature/your-feature-name`
- Bugfix branches: Named as `bugfix/issue-description`

## Review Process

All pull requests will be reviewed by maintainers. We may suggest changes or improvements before merging.

Thank you for contributing to the LLM Model Manager!
