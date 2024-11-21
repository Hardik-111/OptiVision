from flask import Blueprint, render_template, request, jsonify
from app.chatbot_logic import load_fixed_questions, load_log_data

chatbot = Blueprint('chatbot', __name__)

@chatbot.route('/')
def index():
    """Render the chatbot interface."""
    return render_template("index.html")


@chatbot.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot interaction."""
    user_message = request.json.get("message", "")
    response = load_fixed_questions(user_message)
    print("usermessage ",user_message)
    print("response",response)
    if not response:
        response = load_log_data(user_message)
    return jsonify({"response": response})
