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

genre_avg_ratings = merged_exploded.groupby('genres')['average_rating'].mean().reset_index()

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
ratings['n_rating'] = (ratings['rating'] - ratings['rating'].min()) / (ratings['rating'].max() - ratings['rating'].min())

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
# Training RMSE
train_actual = user_item_matrix.values
train_predictions = reconstructed_matrix  # Reconstructed ratings for all users and items

train_rmse = np.sqrt(mean_squared_error(train_actual, train_predictions))
print("Training RMSE:", train_rmse)

# Test RMSE (Already calculated in previous code)
test_actual = test.values
user_ids = test.index
movie_ids = test.columns

# Map users and movies
user_row_indices = [user_id_map.get(user_id, -1) for user_id in user_ids]
movie_col_indices = [movie_id_map.get(movie_id, -1) for movie_id in movie_ids]

# Filter valid indices
user_row_indices = [idx for idx in user_row_indices if idx != -1]
movie_col_indices = [idx for idx in movie_col_indices if idx != -1]

test_predictions = reconstructed_matrix[np.ix_(user_row_indices, movie_col_indices)]
test_rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
print("Test RMSE:", test_rmse)

# Compare RMSE values
if test_rmse > train_rmse * 1.5:  # Arbitrary threshold for overfitting
    print("The model is overfitting.")
else:
    print("The model is not overfitting.")


# %%
ratings.head()

# %%
predicted_ratings.head()

# %% [markdown]
# ## Feature Engineering

# %%
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Overall background and font settings */
        body {
            background-color: #121212;
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        /* Title styling */
        h1, h2 {
            color: #E50914; /* Netflix Red */
        }
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #181818;
        }
        /* Dataframe table styling */
        .dataframe {
            background-color: #282828;
            color: #FFFFFF;
            border: 1px solid #333333;
        }
        /* Buttons and inputs */
        .stButton>button {
            background-color: #E50914;
            color: #FFFFFF;
            border: None;
            border-radius: 5px;
        }
        .stSelectbox>div>div>div {
            color:rgb(239, 234, 234);
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🎬 Movie Recommendation System")

# Sidebar for filters
with st.sidebar:
    st.header("Filters")
    user_id = st.selectbox("Select User ID", ["None"] + user_item_matrix.index.astype(str).tolist())
    selected_genre = st.selectbox("Select Genre", ["None"] + genres_count.index.tolist())
    selected_movie = st.selectbox("Select Movie", ["None"] + movies['title'].unique().tolist())

# Recommendations header
st.subheader("📽️ Top Recommendations")

# Step 1: Prepare all movies with predictions
recommendations = predicted_ratings.reset_index().melt(id_vars="userId", var_name="movieId", value_name="predicted_rating")
recommendations['movieId'] = recommendations['movieId'].astype(int)
recommendations = recommendations.merge(movies, on="movieId", how="left")

# Step 2: Apply filters
if user_id != "None":
    user_id = int(user_id)
    user_ratings = predicted_ratings.loc[user_id]
    recommendations = recommendations[recommendations['movieId'].isin(user_ratings.index)]
    recommendations['predicted_rating'] = user_ratings.loc[recommendations['movieId']].values

if selected_genre != "None":
    recommendations = recommendations[recommendations['genres'].apply(lambda x: selected_genre in x)]

if selected_movie != "None":
    selected_movie_id = movies[movies['title'] == selected_movie]['movieId'].values[0]
    selected_movie_genres = movies[movies['movieId'] == selected_movie_id]['genres'].values[0]
    recommendations = recommendations[recommendations['genres'].apply(lambda x: any(genre in x for genre in selected_movie_genres))]
    recommendations = recommendations[recommendations['movieId'] != selected_movie_id]  # Exclude the selected movie

# Step 3: Ensure diversity in recommendations
recommendations = recommendations.drop_duplicates(subset="movieId").sort_values(by="predicted_rating", ascending=False)

final_recommendations = []
seen_movies = set()
seen_genres = set()

for _, row in recommendations.iterrows():
    genres = row['genres']
    movie_id = row['movieId']
    if movie_id not in seen_movies and not any(genre in seen_genres for genre in genres):
        final_recommendations.append(row)
        seen_movies.add(movie_id)
        seen_genres.update(genres)
    if len(final_recommendations) >= 10:
        break

final_recommendations = pd.DataFrame(final_recommendations)

# Step 4: Display recommendations
if final_recommendations.empty:
    st.write("😔 No recommendations found. Try adjusting the filters.")
else:
    # Use columns for a Netflix-like layout
    st.write("Here are your top picks:")
    cols = st.columns(5)
    for idx, row in final_recommendations.iterrows():
        with cols[idx % 5]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"⭐ Rating: {row['predicted_rating']:.2f}")
            st.markdown(f"🎭 Genre: {row['genres']}")



