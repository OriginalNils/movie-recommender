import streamlit as st
from src.data_loader import download_and_extract_data, load_movie_data
from src.collaborative_filtering import SVDRecommender

# --- Page Configuration ---
st.set_page_config(page_title="Movie Recommender", layout="centered")

# --- Caching the model ---
# Use Streamlit's cache to load data and train the model only once
@st.cache_resource
def load_and_train_model():
    """
    Loads data and trains the SVD recommender model.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    print("Loading data and training model for the first time...")
    download_and_extract_data()
    movies, ratings = load_movie_data()
    
    # Train the final model on all data with the best k
    recommender = SVDRecommender(ratings, movies)
    recommender.fit(k=20) # Using the optimal k=20
    
    return recommender, movies, ratings

# --- Main App ---
st.title("🎬 Movie Recommendation Engine")
st.write("This app recommends movies to users based on the MovieLens dataset.")

# Load the trained model (will be cached after the first run)
recommender, movies, ratings = load_and_train_model()

# --- User Input ---
st.header("Get Your Recommendations")

# Create a list of all user IDs
user_ids = ratings['userId'].unique().tolist()
selected_user_id = st.selectbox("Choose a User ID", user_ids)

if st.button("Recommend Movies"):
    if selected_user_id:
        try:
            # Get and display recommendations
            recommendations = recommender.recommend(selected_user_id, num_recommendations=10)
            
            st.subheader(f"Top 10 Recommendations for User {selected_user_id}:")
            
            # Display recommendations in a cleaner format
            for index, row in recommendations.iterrows():
                st.write(f"**{row['title']}** ({row['genres']})")

        except Exception as e:
            st.error(f"An error occurred: {e}")
