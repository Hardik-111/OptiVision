from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-fallback-key')

    # Import routes
    from app.routes import chatbot
    app.register_blueprint(chatbot)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='localhost', port=3000, debug=True)