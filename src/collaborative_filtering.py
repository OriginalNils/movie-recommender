import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler

class SVDRecommender:
    """
    A recommendation engine using Singular Value Decomposition (SVD).
    """
    def __init__(self, ratings_df, movies_df):
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.user_item_matrix = self._create_user_item_matrix()
        self.scaler = MinMaxScaler()
        self.predictions_df = None

    def _create_user_item_matrix(self):
        """Creates the user-item matrix from the ratings data."""
        return self.ratings_df.pivot_table(index='userId', columns='movieId', values='rating')

    def fit(self, k=50):
        """
        Trains the SVD model using a robust demeaning method.
        """
        print("Training the SVD model...")
        matrix = self.user_item_matrix
        
        # 1. Calculate the mean rating for each user, ignoring NaNs.
        user_ratings_mean = matrix.mean(axis=1)
        
        # 2. Subtract the mean from the known ratings only.
        # The .subtract() method correctly handles the alignment. Unrated (NaN) values remain NaN.
        matrix_demeaned = matrix.subtract(user_ratings_mean, axis=0)
        
        # 3. Fill the remaining NaNs (unrated movies) with 0.
        # Now, rated movies are centered around 0, and unrated movies are exactly 0.
        matrix_demeaned_filled = matrix_demeaned.fillna(0)
        
        # 4. Perform SVD on this correctly prepared matrix.
        R = matrix_demeaned_filled.values
        U, sigma, Vt = svds(R, k=k)
        sigma_diag_matrix = np.diag(sigma)
        
        # 5. Reconstruct the demeaned ratings matrix.
        all_user_predicted_ratings_demeaned = np.dot(np.dot(U, sigma_diag_matrix), Vt)
        
        # 6. Add the user's mean rating back to get the final predicted ratings.
        all_user_predicted_ratings = all_user_predicted_ratings_demeaned + user_ratings_mean.values.reshape(-1, 1)
        
        self.predictions_df = pd.DataFrame(all_user_predicted_ratings, columns=matrix.columns, index=matrix.index)
        print("Model training complete.")

    def recommend(self, user_id, num_recommendations=10):
        """
        Recommends movies for a given user.
        """
        if self.predictions_df is None:
            raise RuntimeError("You must train the model first by calling the .fit() method.")
            
        sorted_user_predictions = self.predictions_df.loc[user_id].sort_values(ascending=False)
        user_rated_movies = self.ratings_df[self.ratings_df.userId == user_id]
        unseen_movies = self.movies_df[~self.movies_df['movieId'].isin(user_rated_movies['movieId'])]
        
        recommendations = unseen_movies.merge(
            pd.DataFrame(sorted_user_predictions).reset_index(), how='left', on='movieId'
        ).rename(columns={user_id: 'prediction'}).sort_values('prediction', ascending=False)
        
        return recommendations.head(num_recommendations)