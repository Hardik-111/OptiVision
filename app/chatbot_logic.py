import pandas as pd
import datetime
import re
from collections import Counter

# Load the CSV log data
def load_log_data(csv_file):
    return pd.read_csv('detection_log.csv')

# Load fixed questions and answers from a CSV
def load_fixed_questions(csv_file):
    fixed_qa = pd.read_csv('fixed_qa.csv')
    return dict(zip(fixed_qa['Question'].str.lower(), fixed_qa['Answer']))

# Function to parse time and date
def parse_datetime(row):
    date_str = row['date']
    time_str = row['time']
    return datetime.datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %I:%M:%S %p")

# Preprocess the log data
def preprocess_log_data(log_data):
    log_data['datetime'] = log_data.apply(parse_datetime, axis=1)
    return log_data

# Function to count the number of people in a time range
def count_people_in_time_range(log_data, start_time, end_time):
    start_time = datetime.datetime.strptime(start_time, "%d-%m-%Y %I:%M:%S %p")
    end_time = datetime.datetime.strptime(end_time, "%d-%m-%Y %I:%M:%S %p")
    
    filtered_data = log_data[(log_data['datetime'] >= start_time) & (log_data['datetime'] <= end_time)]
    return filtered_data[filtered_data['class'] == 'person'].shape[0]

# Function to find usual/max entry times
def entry_time_statistics(log_data):
    entry_times = log_data[log_data['class'] == 'person']['time']
    time_counts = Counter(entry_times)
    most_common_time = time_counts.most_common(1)[0]
    return most_common_time

# Function to find usual exit time
def exit_time_statistics(log_data):
    exit_times = log_data[log_data['class'] == 'person']['time']
    time_counts = Counter(exit_times)
    most_common_exit_time = time_counts.most_common(1)[0]
    return most_common_exit_time

# Function to find when people arrive in flocks
def flock_arrival_times(log_data, flock_threshold=5):
    time_counts = Counter(log_data['time'])
    flock_times = {time: count for time, count in time_counts.items() if count >= flock_threshold}
    return flock_times

# Chatbot function to process the user's query
def chatbot(log_data, fixed_qa, query):
    query = query.lower()

    if "number of people between" in query:
        times = re.findall(r'\d{2}-\d{2}-\d{4} \d{1,2}:\d{2}:\d{2} [apm]{2}', query)
        if len(times) == 2:
            count = count_people_in_time_range(log_data, times[0], times[1])
            return f"The number of people detected between {times[0]} and {times[1]} is {count}."
    
    elif "entry time statistics" in query:
        most_common_time, count = entry_time_statistics(log_data)
        return f"The most common entry time is {most_common_time} with {count} entries."

    elif "exit time statistics" in query:
        most_common_exit_time, count = exit_time_statistics(log_data)
        return f"The most common exit time is {most_common_exit_time} with {count} exits."

    elif "people arrive in flocks" in query:
        flocks = flock_arrival_times(log_data)
        if flocks:
            return f"People arrived in flocks at these times: {', '.join([f'{time} ({count} people)' for time, count in flocks.items()])}."
        else:
            return "No flock arrivals detected."

    # Fixed Q&A
    elif query in fixed_qa:
        return fixed_qa[query]

    else:
        return "Sorry, I don't know the answer to that question."


# Main function to simulate chatbot interaction
def main():
    # Load the CSV files
    log_data = load_log_data('detection_log.csv')  # Replace with actual log CSV file path
    fixed_qa = load_fixed_questions('fixed_qa.csv')  # Replace with fixed Q&A CSV file path
    
    # Preprocess log data
    log_data = preprocess_log_data(log_data)

    print("Chatbot is ready! Type 'exit' to quit.")
    
    while True:
        query = input("Ask me a question: ").strip()
        if query.lower() == "exit":
            break
        
        response = chatbot(log_data, fixed_qa, query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
