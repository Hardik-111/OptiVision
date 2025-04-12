from flask import Flask
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-fallback-key')

    # Register Blueprints
    from app.routes import chatbot
    app.register_blueprint(chatbot)

    return app