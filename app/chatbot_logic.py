import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE = "unmatched_queries.log"

def log_unmatched_query(query):
    """
    Log unmatched queries to a file for review.
    """
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{query}\n")

class FixedResponseHandler:
    """
    Handle fixed Q&A responses using exact and fuzzy matching.
    """
    def __init__(self, csv_file):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"The file {csv_file} was not found.")
        self.qa_dict, self.fixed_qa_df = self.load_fixed_question(csv_file)

        # Initialize TF-IDF Vectorizer once and fit it to all questions
        self.vectorizer = TfidfVectorizer(stop_words='english')  # Adding stop words removal
        self.vectorizer.fit(self.fixed_qa_df['Question'].tolist())
        
    def load_fixed_question(self, csv_file):
        """
        Load fixed Q&A pairs from a CSV file.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"The file {csv_file} was not found. Please check the path.")
        
        # Read the CSV file into a DataFrame
        fixed_qa_df = pd.read_csv(csv_file)
        fixed_qa_df['Question'] = fixed_qa_df['Question'].str.lower().str.strip()  # Normalize questions
        qa_dict = dict(zip(fixed_qa_df['Question'], fixed_qa_df['Answer']))
        logging.info("Loaded fixed Q&A DataFrame successfully.")
        return qa_dict, fixed_qa_df

    def get_fixed_response(self, query, threshold=0.5):
        query = query.lower().strip()  # Ensure query is stripped of leading/trailing spaces
        
        # Exact match first
        if query in self.qa_dict:
            return self.qa_dict[query]
        
        # Use TF-IDF and cosine similarity for fuzzy matching
        query_vec = self.vectorizer.transform([query])
        question_vecs = self.vectorizer.transform(self.fixed_qa_df['Question'])
        similarities = cosine_similarity(query_vec, question_vecs).flatten()

        # Get the closest match
        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]

        # Print debug information for similarity comparison
        logging.debug(f"Query: '{query}'")
        logging.debug(f"Best match: '{self.fixed_qa_df['Question'].iloc[best_match_idx]}' with score: {best_match_score}")

        if best_match_score >= threshold:
            best_match_question = self.fixed_qa_df['Question'].iloc[best_match_idx]
            return self.qa_dict[best_match_question]
        else:
            logging.info(f"Query not matched: {query}")  # Log unmatched queries
            return "Sorry, I couldn't find an answer. Please try a different query."  # Return empty string instead of fallback message



class DynamicResponseHandler:
    """
    A class to dynamically respond to queries based on detection data loaded from a CSV file.
    """
    def __init__(self, csv_file):
        """
        Initialize the handler by loading dynamic responses and detection data.
        """
        self.dynamic_responses, self.detection_data = self.load_dynamic_responses(csv_file)

    def load_dynamic_responses(self, csv_file):
        """
        Load dynamic responses and detection data from a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            tuple: A dictionary of dynamic response functions and a DataFrame of detection data.
        """
        if not os.path.exists(csv_file):
            logging.warning(f"Dynamic responses file {csv_file} not found. Proceeding without it.")
            return {}, None

        detection_data = pd.read_csv(csv_file)
        dynamic_responses = {
            'max people count': self.get_max_people_count,
            'current people count': self.get_current_people_count,
            'last detection time': self.get_last_detection_time,
            'last detected object': self.get_last_detected_object,
            'detection details': self.get_detection_details,
            'detection trends': self.get_detection_trends,
            'tracking details': self.get_tracking_details,
            'first detected object': self.get_first_detected_object,
            'first detection time': self.get_first_detection_time,
            'last detected object and time': self.get_last_detected_object_and_time,
            'detection summary': self.get_detection_summary
        }
        logging.info("Loaded dynamic responses and detection data successfully.")
        return dynamic_responses, detection_data

    def get_max_people_count(self, query):
        """
        Return the maximum people count detected and its associated time.
        """
        try:
            max_count_row = self.detection_data.loc[self.detection_data['max_people_count'].idxmax()]
            max_people_count = max_count_row['max_people_count']
            max_time = max_count_row['max_time']
            return f"The maximum people count detected was {max_people_count} at {max_time}."
        except Exception as e:
            logging.error(f"Error getting max people count: {e}")
            return "Sorry, I couldn't fetch the maximum people count."

    def get_current_people_count(self, query):
        """
        Return the current people count.
        """
        try:
            current_count = self.detection_data.iloc[-1]['current_people_count']
            return f"The current number of people detected is {current_count}."
        except Exception as e:
            logging.error(f"Error getting current people count: {e}")
            return "Sorry, I couldn't fetch the current people count."

    def get_last_detection_time(self, query):
        """
        Return the time of the last detection.
        """
        try:
            last_detection_time = self.detection_data.iloc[-1]['Time (IST)']
            return f"The last detection occurred at {last_detection_time}."
        except Exception as e:
            logging.error(f"Error getting last detection time: {e}")
            return "Sorry, I couldn't fetch the time of the last detection."

    def get_last_detected_object(self, query):
        """
        Return the last detected object.
        """
        try:
            last_object = self.detection_data.iloc[-1]['Class Name']
            return f"The last detected object was a {last_object}."
        except Exception as e:
            logging.error(f"Error getting last detected object: {e}")
            return "Sorry, I couldn't fetch the last detected object."

    def get_detection_details(self, query):
        """
        Return details of the last detection.
        """
        try:
            last_row = self.detection_data.iloc[-1]
            return f"The last detection was a {last_row['Class Name']} on {last_row['Date']} at {last_row['Time (IST)']} with dimensions {last_row['Dimensions (Width x Height)']}."
        except Exception as e:
            logging.error(f"Error getting detection details: {e}")
            return "Sorry, I couldn't fetch the detection details."

    def get_detection_time(self, query):
        """
        Return the time of the last detection.
        """
        try:
            last_detection_time = self.detection_data.iloc[-1]['Time (IST)']
            return f"The time of the latest detection was {last_detection_time}."
        except Exception as e:
            logging.error(f"Error getting detection time: {e}")
            return "Sorry, I couldn't fetch the time of the latest detection."

    def get_tracking_details(self, query):
        """
        Return tracking details of detected people.
        """
        try:
            last_row = self.detection_data.iloc[-1]
            tracking_details = f"Tracking details:\n" \
                               f"Class: {last_row['Class Name']}\n" \
                               f"Date: {last_row['Date']}\n" \
                               f"Time: {last_row['Time (IST)']}\n" \
                               f"Dimensions: {last_row['Dimensions (Width x Height)']}\n" \
                               f"Xmin: {last_row['Xmin']}, Ymin: {last_row['Ymin']}, Xmax: {last_row['Xmax']}, Ymax: {last_row['Ymax']}\n" \
                               f"Frame Width: {last_row['Frame Width']}, Frame Height: {last_row['Frame Height']}"
            return tracking_details
        except Exception as e:
            logging.error(f"Error getting tracking details: {e}")
            return "Sorry, I couldn't fetch the tracking details."

    def get_first_detected_object(self, query):
        """
        Return the first detected object.
        """
        try:
            first_object = self.detection_data.iloc[0]['Class Name']
            return f"The first detected object was a {first_object}."
        except Exception as e:
            logging.error(f"Error getting first detected object: {e}")
            return "Sorry, I couldn't fetch the first detected object."

    def get_first_detection_time(self, query):
        """
        Return the time of the first detection.
        """
        try:
            first_detection_time = self.detection_data.iloc[0]['Time (IST)']
            return f"The first detection occurred at {first_detection_time}."
        except Exception as e:
            logging.error(f"Error getting first detection time: {e}")
            return "Sorry, I couldn't fetch the time of the first detection."

    def get_last_detected_object_and_time(self, query):
        """
        Return the last detected object and its detection time.
        """
        try:
            last_row = self.detection_data.iloc[-1]
            last_object = last_row['Class Name']
            last_time = last_row['Time (IST)']
            return f"The last detected object was a {last_object} at {last_time}."
        except Exception as e:
            logging.error(f"Error getting last detected object and time: {e}")
            return "Sorry, I couldn't fetch the last detected object and time."

    def get_detection_summary(self, query):
        """
        Provide a summary of the most recent detection.
        """
        try:
            last_row = self.detection_data.iloc[-1]
            summary = f"Detection Summary:\n" \
                      f"Object: {last_row['Class Name']}\n" \
                      f"Date: {last_row['Date']}\n" \
                      f"Time: {last_row['Time (IST)']}\n" \
                      f"Dimensions: {last_row['Dimensions (Width x Height)']}."
            return summary
        except Exception as e:
            logging.error(f"Error getting detection summary: {e}")
            return "Sorry, I couldn't fetch the detection summary."

    def get_detection_trends(self, query):
        """
        Provide detection trends (e.g., changes in people count over time).
        """
        try:
            trends = self.detection_data[['Date', 'Time (IST)', 'current_people_count']]
            trends_str = "Detection Trends:\n"
            for index, row in trends.iterrows():
                trends_str += f"On {row['Date']} at {row['Time (IST)']}, the people count was {row['current_people_count']}.\n"
            return trends_str
        except Exception as e:
            logging.error(f"Error getting detection trends: {e}")
            return "Sorry, I couldn't fetch the detection trends."

    def get_dynamic_response(self, query):
        """
        Return a dynamic response based on the query.
        """
        query = query.lower()
        for key, response_func in self.dynamic_responses.items():
            if key in query:
                return response_func(query)
        return "Sorry, I couldn't find an answer. Please try a different query."

