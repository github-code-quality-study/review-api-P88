import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, parse_qsl, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        self.allowed_locations = {
        "Albuquerque, New Mexico",
        "Carlsbad, California",
        "Chula Vista, California",
        "Colorado Springs, Colorado",
        "Denver, Colorado",
        "El Cajon, California",
        "El Paso, Texas",
        "Escondido, California",
        "Fresno, California",
        "La Mesa, California",
        "Las Vegas, Nevada",
        "Los Angeles, California",
        "Oceanside, California",
        "Phoenix, Arizona",
        "Sacramento, California",
        "Salt Lake City, Utah",
        "San Diego, California",
        "Tucson, Arizona"
        }

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores
    def filter_reviews(self,location:str,start_date:str, end_date:str):
        filtered_reviews = []
        
        if location and location not in self.allowed_locations:
            return filtered_reviews
        for review in reviews:
            review_location = review["Location"]
            review_timestamp = datetime.strptime(review["Timestamp"],'%Y-%m-%d %H:%M:%S').date()
            if location and location != review_location:
                continue
            if start_date:
                start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d').date()
                if review_timestamp < start_date_parsed:
                    continue
            if end_date:
                end_date_parsed = datetime.strptime(end_date, '%Y-%m-%d').date()
                if review_timestamp > end_date_parsed:
                    continue
            sentiment = self.analyze_sentiment(review["ReviewBody"])
            filtered_reviews.append({
            "ReviewId": review["ReviewId"],
            "Location": review["Location"],
            "ReviewBody": review["ReviewBody"],
            "Timestamp": review["Timestamp"],
            "sentiment": sentiment
        })
        filtered_reviews.sort(key=lambda r: r["sentiment"]["compound"], reverse=True)
        return filtered_reviews

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # parse query params 
            raw_query_string = environ.get("QUERY_STRING", "") 
            # Parse the query string
            parsed_url = urlparse(f"/?{raw_query_string}")
            query_params_list = parse_qsl(parsed_url.query)
            query_params = dict(query_params_list)  
            location = query_params.get("location", None)
            start_date = query_params.get("start_date", None)
            end_date = query_params.get("end_date", None)

            # Create the response body from the reviews and convert to a JSON byte string
            response_data = self.filter_reviews(location, start_date, end_date)
            response_body = json.dumps(response_data, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            try:
                # Parse POST request data
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size).decode("utf-8")
                post_params = parse_qs(request_body)

                location = post_params.get("Location", [None])[0]
                review_body = post_params.get("ReviewBody", [None])[0]
                if location not in self.allowed_locations:
                    raise ValueError("Invalid location.")

                if not location or not review_body:
                    raise ValueError("Location and ReviewBody are required fields.")

                # Create new review
                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            except Exception as e:
                error_response = {"error": str(e)}
                response_body = json.dumps(error_response, indent=2).encode("utf-8")

                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()