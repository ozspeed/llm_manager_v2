# Entry point for LLM Manager v2
from flask import Flask
from config.settings import get_config

def create_app():
    app = Flask(__name__)
    config = get_config()
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    # Blueprints and further config will be registered here
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=app.config['DEBUG'])
