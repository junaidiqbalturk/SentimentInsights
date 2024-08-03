from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "greamesmith"

    from .routes import main
    app.register_blueprint(main)

    return app

