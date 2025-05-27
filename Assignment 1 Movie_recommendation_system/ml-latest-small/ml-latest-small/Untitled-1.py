# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error


# %%
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv') 

# %%
print(links.shape)
links.head()

# %%
print(movies.shape)
movies.head()

# %%
print(ratings.shape)
ratings.head()

# %%
print(tags.shape)
tags.head()

# %%
ratings.describe()

# %% [markdown]
# ## EDA

# %%
# 1. Ratings Distribution
sns.histplot(ratings['rating'], bins=10, kde=False, color='blue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# %%
# 2. Top 10 Movies with the Most Ratings
top_movies = ratings['movieId'].value_counts().head(10).index
top_movie_titles = movies[movies['movieId'].isin(top_movies)]
top_movie_ratings = ratings[ratings['movieId'].isin(top_movies)]['movieId'].value_counts()

plt.bar(top_movie_titles['title'], top_movie_ratings)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Movies by Number of Ratings')
plt.ylabel('Number of Ratings')
plt.xlabel('Movie Titles')
plt.show()

# %%
movies['genres'] = movies['genres'].str.split('|')
genres_expanded = movies.explode('genres')
genres_count = genres_expanded['genres'].value_counts()

sns.barplot(x=genres_count.index, y=genres_count.values, palette='viridis')
plt.title('Number of Movies per Genre')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.xlabel('Genres')
plt.show()

# %%
# Calculate Average Rating
avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'average_rating']
avg_ratings.head()

# %%
ratings.head()

# %%
merged = pd.merge(avg_ratings,movies)
merged.head()

# %%
merged_exploded = merged.explode('genres')

# Calculate average ratings for each genre
genre_avg_ratings = merged_exploded.groupby('genres')['average_rating'].mean().reset_index()

# Plot the average rating per genre
plt.figure(figsize=(10, 6))
sns.barplot(data=genre_avg_ratings, x='genres', y='average_rating', palette='coolwarm')
plt.title('Average Rating per Genre')
plt.xlabel('Genres')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.show()


# %% [markdown]
# ## Preprocessing

# %%
# Handling the missing values 
links.isnull().sum()


# %%
movies.isnull().sum()

# %%
ratings.isnull().sum()

# %%
tags.isnull().sum()

# %%
# Step 1: Create the user-item interaction matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Step 2: Fill missing values (e.g., with 0 for no interaction or the mean rating for imputation)
user_item_matrix_filled = user_item_matrix.fillna(0)  # Replace NaNs with 0 (no interaction)

# Display the user-item interaction matrix
print("User-Item Interaction Matrix:")
user_item_matrix_filled.head()


# %%
user_item_matrix = user_item_matrix.fillna(0)  # Replace NaNs with zeros


# %%
user_means = user_item_matrix.mean(axis=1)
normalized_matrix = user_item_matrix.sub(user_means, axis=0)
normalized_matrix.head()

# %%
from scipy.sparse.linalg import svds

# Perform SVD, keeping k latent factors
U, sigma, Vt = svds(normalized_matrix.values, k=100)

# Convert sigma (singular values) into a diagonal matrix
sigma = np.diag(sigma)


# %%
reconstructed_matrix = np.dot(np.dot(U, sigma), Vt)

# Add back user means to denormalize
reconstructed_matrix = reconstructed_matrix + user_means.values[:, np.newaxis]
reconstructed_matrix = np.clip(reconstructed_matrix, 0, 5)



# %%
predicted_ratings = pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
predicted_ratings.head()

# %% [markdown]
# ## Predictions

# %%
user_id = 1  # Example user ID


# %%
user_ratings = predicted_ratings.loc[user_id]
ranked_movies = user_ratings.sort_values(ascending=False)


# %%
watched_movies = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
recommendations = ranked_movies.drop(index=watched_movies)


# %%
recommendations = recommendations.reset_index().merge(movies, on="movieId", how="left")


# %%
print(recommendations.columns)


# %%
recommendations.rename(columns={1: 'predicted_rating'}, inplace=True)
print(recommendations[['movieId', 'title', 'predicted_rating']].head(10))


# %% [markdown]
# ## Evaluation

# %%
train, test = train_test_split(user_item_matrix, test_size=0.2, random_state=42)

# %%
# Assuming user_item_matrix is your original DataFrame
# Get the user and movie indices
user_ids = test.index
movie_ids = test.columns

# Ensure that user_ids and movie_ids are within the reconstructed matrix dimensions
user_id_map = {user_id: idx for idx, user_id in enumerate(user_item_matrix.index)}
movie_id_map = {movie_id: idx for idx, movie_id in enumerate(user_item_matrix.columns)}

# Map the user and movie ids to their respective row and column indices in reconstructed_matrix
user_row_indices = [user_id_map[user_id] for user_id in user_ids if user_id in user_id_map]
movie_col_indices = [movie_id_map[movie_id] for movie_id in movie_ids if movie_id in movie_id_map]

# Use the indices to get the predicted ratings from the reconstructed matrix
test_predictions = reconstructed_matrix[np.ix_(user_row_indices, movie_col_indices)]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test.values, test_predictions))
print("RMSE:", rmse)


# %%
# import streamlit as st

# st.title("Movie Recommendation System")

# # Select user
# user_id = st.selectbox("Select User ID:", user_item_matrix.index)

# # Get recommendations
# user_ratings = predicted_ratings.loc[user_id]
# ranked_movies = user_ratings.sort_values(ascending=False)
# recommendations = ranked_movies.reset_index().merge(movies, on="movieId", how="left")

# # Display recommendations
# st.write("Top Recommendations:")
# st.dataframe(recommendations[['title']].head(10))


# %%
ratings.head()

# %%
predicted_ratings.head()

# %%
import streamlit as st
import requests
import urllib.request
from urllib.parse import quote
# OMDB API Key
OMDB_API_KEY = "99587fce&"

def fetch_poster(title):
    """Fetch movie poster URL from OMDB API."""
    try:
        response = requests.get(f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={quote(title)}")
        data = response.json()
        if data.get("Poster"):
            return data["Poster"]
        return "https://via.placeholder.com/150"  # Placeholder if poster not available
    except Exception as e:
        print("ERROR: " + str(e))
        return "https://via.placeholder.com/150"

# Sidebar filters
st.sidebar.header("Filters")
user_id = st.sidebar.selectbox("User ID :", ["None"] + user_item_matrix.index.astype(str).tolist())
selected_genre = st.sidebar.selectbox("Genre :", ["None"] + genres_count.index.tolist())

st.subheader("Movie Recommendation System")

# Generate recommendations
recommendations = predicted_ratings.reset_index().melt(id_vars="userId", var_name="movieId", value_name="predicted_rating")
recommendations['movieId'] = recommendations['movieId'].astype(int)
recommendations = recommendations.merge(movies, on="movieId", how="left")

if user_id != "None":
    user_id = int(user_id)
    recommendations = recommendations[recommendations['userId'] == user_id]

if selected_genre != "None":
    recommendations = recommendations[recommendations['genres'].apply(lambda x: selected_genre in x)]

recommendations = recommendations.sort_values(by="predicted_rating", ascending=False).head(10)

if recommendations.empty:
    st.write("No recommendations found. Try adjusting the filters.")
else:
    for _, row in recommendations.iterrows():
        poster_url = fetch_poster(row['title'])
        st.image(poster_url, width=300)
        st.write(f"**{row['title']}**")
        st.write(f"Predicted Rating: {row['predicted_rating']:.2f}")
        st.write(f"Genres: {', '.join(row['genres'])}")
        st.write("---")


