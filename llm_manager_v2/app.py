# Entry point for LLM Manager v2
from flask import Flask
from config.settings import get_config

from flask import render_template

def create_app():
    app = Flask(
        __name__,
        template_folder="views/templates",
        static_folder="static"
    )
    config = get_config()
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG

    # Register Blueprints
    from controllers.huggingface_controller import huggingface_bp
    from controllers.settings_controller import settings_bp
    from controllers.models_controller import models_bp
    app.register_blueprint(huggingface_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(models_bp)

    # View routes

    @app.route("/huggingface")
    def huggingface_view():
        return render_template("huggingface.html")

    @app.route("/settings")
    def settings_view():
        return render_template("settings.html")

    @app.route("/")
    @app.route("/models")
    def models_view():
        return render_template("models.html")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=app.config['DEBUG'])
