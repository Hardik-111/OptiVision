from flask import Blueprint, render_template, request, jsonify
from app.chatbot_logic import load_fixed_questions, load_fixed_response

chatbot = Blueprint('chatbot', __name__)

# Load fixed Q&A data when the application starts
qa_dict = load_fixed_questions("fixed_qa.csv")

@chatbot.route('/')
def index():
    """Render the chatbot interface."""
    return render_template("index.html")

@chatbot.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot interaction."""
    user_message = request.json.get("message", "")
    print("User message:", user_message)  # Debug
    response = load_fixed_response(user_message, qa_dict)
    print("Response:", response)  # Debug
    return jsonify({"response": response})

@chatbot.route('/reload', methods=['GET'])
def reload_qa():
    global qa_dict
    qa_dict = load_fixed_questions("fixed_qa.csv")
    return jsonify({"message": "Q&A data reloaded successfully!"})
