import os
import pandas as pd
import requests
import zipfile
from io import BytesIO

# The URL for the small MovieLens dataset (100k ratings)
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data" # The directory to store the data

def download_and_extract_data():
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        print(f"Creating directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)

    # Check if the data is already downloaded
    movies_file = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    if os.path.exists(movies_file):
        print("MovieLens dataset already exists. Skipping download.")
        return

    print(f"Downloading dataset from {DATASET_URL}...")
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status() # Raise an exception for bad status codes

        # Unzip the file in memory
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            print("Extracting files...")
            z.extractall(DATA_DIR)
        print("Dataset downloaded and extracted successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the dataset: {e}")


def load_movie_data() -> (pd.DataFrame, pd.DataFrame):
    # Define the paths to the csv files
    movies_path = os.path.join(DATA_DIR, "ml-latest-small", "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(
            "Dataset files not found. Please run download_and_extract_data() first."
        )

    # Load the data
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    
    return movies_df, ratings_df