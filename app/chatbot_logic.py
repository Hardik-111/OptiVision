import pandas as pd
import os

# Function to load fixed Q&A from a CSV
def load_fixed_questions(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"The file {csv_file} was not found. Please check the path.")
    
    # Read the CSV file into a DataFrame
    fixed_qa_df = pd.read_csv(csv_file)
    print("Loaded fixed Q&A DataFrame:\n", fixed_qa_df)  # Debug: Print the DataFrame
    
    # Convert to dictionary: questions as keys, answers as values
    qa_dict = dict(zip(fixed_qa_df['Question'].str.lower(), fixed_qa_df['Answer']))
    print("Processed Q&A dictionary:\n", qa_dict)  # Debug: Print the dictionary
    
    return qa_dict

# Function to handle user queries
def chatbot(qa_dict, query):
    query = query.lower()
    if query in qa_dict:
        return qa_dict[query]
    else:
        return "Sorry, I don't know the answer to that question."

# Function to directly fetch response for Flask
def load_fixed_response(query, qa_dict):
    query = query.lower()
    print("qa_dict:", qa_dict)  # Debug: Ensure qa_dict is passed and visible
    if query in qa_dict:
        return qa_dict[query]
    else:
        return "Sorry, I don't know the answer to that question."
