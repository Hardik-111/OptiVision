from flask import Blueprint, render_template, request, jsonify
from app.chatbot_logic import FixedResponseHandler, DynamicResponseHandler

chatbot = Blueprint('chatbot', __name__)

# Initialize handlers for fixed and dynamic responses
fixed_handler = FixedResponseHandler("fixed_qa.csv")
dynamic_handler = DynamicResponseHandler("detection_log2.csv")

@chatbot.route('/')
def index():
    """
    Render the chatbot interface.
    """
    return render_template("index.html")

@chatbot.route('/chat', methods=['POST'])
def chat():
    """
    Handle chatbot interaction by responding to user messages.
    """
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please provide a valid message."}), 400

    print("User message received:", user_message)  # Debug

    # Get both dynamic and fixed responses
    dynamic_response = dynamic_handler.get_dynamic_response(user_message)
    fixed_response = fixed_handler.get_fixed_response(user_message)

    # Check for the presence of "Sorry" in both responses
    if "Sorry" in dynamic_response and "Sorry" not in fixed_response:
        response = fixed_response  # Return fixed response if dynamic has "Sorry"
    elif "Sorry" not in dynamic_response and "Sorry" in fixed_response:
        response = dynamic_response  # Return dynamic response if fixed has "Sorry"
    elif "Sorry" in dynamic_response and "Sorry" in fixed_response:
        response = "Sorry, I couldn't find an answer. Please try a different query."  # Both have "Sorry"
    else:
        response = dynamic_response  # Return dynamic response if both are valid

    print("Response generated:", response)  # Debug
    return jsonify({"response": response})
    


@chatbot.route('/reload', methods=['GET'])
def reload_data():
    """
    Reload both fixed and dynamic data dynamically.
    """
    try:
        fixed_handler.qa_dict, fixed_handler.fixed_qa_df = fixed_handler.load_fixed_question("fixed_qa.csv")
        dynamic_handler.dynamic_responses, dynamic_handler.detection_data = dynamic_handler.load_dynamic_responses("detection_log2.csv")
        message = "Data reloaded successfully!"
        logging.info(message)
        return jsonify({"message": message})
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found error: {fnf_error}")
        return jsonify({"message": f"File not found: {str(fnf_error)}"}), 400
    except Exception as e:
        logging.error(f"Error during data reload: {e}")
        return jsonify({"message": f"Failed to reload data: {str(e)}"}), 500



