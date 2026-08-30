# ============================================================
# CONTENT-BASED FILTERING
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# 1. LOAD DATA
# ============================================================

ratings = pd.read_csv("data/processed/ratings_clean.csv")
movies = pd.read_csv("data/processed/movies_clean.csv")


# ============================================================
# 2. SHARED TRAIN / TEST SPLIT
#    Use the SAME settings for all group members
# ============================================================

train, test = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

# Same 8,000 ratings for RMSE and MAE
evaluation_8000 = (
    test.groupby("userId")
        .filter(lambda x: len(x) >= 3)
        .sample(n=8000, random_state=42)
)

# Same 100 users for Top-K evaluation
topk_users = (
    test.groupby("userId")
        .filter(lambda x: len(x) >= 5)["userId"]
        .drop_duplicates()
        .head(100)
        .tolist()
)

RELEVANCE_THRESHOLD = 3.5
K = 10


print("Shared evaluation settings")
print("--------------------------")
print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")
print(f"RMSE/MAE records: {len(evaluation_8000):,}")
print(f"Top-K users: {len(topk_users)}")
print(f"Relevant rating: >= {RELEVANCE_THRESHOLD}")
print(f"K: {K}")


# ============================================================
# 3. CONTENT-BASED FEATURE PREPARATION
# ============================================================

content_features = [
    "genres_str",
    "keyword_list",
    "overview",
    "cast",
    "original_language"
]

# Handle missing values
for feature in content_features:
    movies[feature] = (
        movies[feature]
        .fillna("")
        .astype(str)
    )

# Combine movie features
movies["content"] = (
    movies["genres_str"] + " " +
    movies["keyword_list"] + " " +
    movies["overview"] + " " +
    movies["cast"] + " " +
    movies["original_language"]
)

# Reset movie index
movies = movies.reset_index(drop=True)


# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000
)

tfidf_matrix = tfidf.fit_transform(
    movies["content"]
)

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)


# ============================================================
# 5. COSINE SIMILARITY
# ============================================================

cosine_sim = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

print("Cosine similarity matrix shape:", cosine_sim.shape)


# ============================================================
# 6. MOVIE INDEX
# ============================================================

movie_indices = pd.Series(
    movies.index,
    index=movies["title"].str.lower()
).drop_duplicates()


# ============================================================
# 7. CONTENT-BASED RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(title, num_recommendations=10):

    title_lower = title.lower()

    if title_lower not in movie_indices:
        return pd.DataFrame()

    idx = movie_indices[title_lower]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the input movie itself
    similarity_scores = similarity_scores[
        1:num_recommendations + 1
    ]

    movie_indices_result = [
        item[0] for item in similarity_scores
    ]

    scores = [
        item[1] for item in similarity_scores
    ]

    recommendations = movies.iloc[
        movie_indices_result
    ][
        ["tmdbId", "title"]
    ].copy()

    recommendations["similarity"] = scores

    return recommendations.reset_index(drop=True)


# ============================================================
# 8. CONTENT-BASED RATING PREDICTION
#    Used for RMSE and MAE
# ============================================================

def predict_rating_content_based(
    user_id,
    movie_id,
    train_data
):

    # User's previous ratings
    user_history = train_data[
        train_data["userId"] == user_id
    ]

    if user_history.empty:
        return np.nan

    # Find target movie
    target_index = movies.index[
        movies["tmdbId"] == movie_id
    ]

    if len(target_index) == 0:
        return np.nan

    target_index = target_index[0]

    # Similarity between target movie and all movies
    similarities = cosine_sim[target_index]

    weighted_ratings = []
    similarity_values = []

    for _, row in user_history.iterrows():

        rated_movie_id = row["tmdbId"]

        rated_index = movies.index[
            movies["tmdbId"] == rated_movie_id
        ]

        if len(rated_index) == 0:
            continue

        rated_index = rated_index[0]

        similarity = similarities[rated_index]

        if similarity <= 0:
            continue

        weighted_ratings.append(
            similarity * row["rating"]
        )

        similarity_values.append(similarity)

    if len(similarity_values) == 0:
        return np.nan

    predicted_rating = (
        sum(weighted_ratings) /
        sum(similarity_values)
    )

    # Keep prediction within MovieLens rating scale
    predicted_rating = np.clip(
        predicted_rating,
        0.5,
        5.0
    )

    return predicted_rating


# ============================================================
# 9. RMSE AND MAE
#    Uses the SAME evaluation_8000 as other models
# ============================================================

predictions = []

for _, row in evaluation_8000.iterrows():

    predicted = predict_rating_content_based(
        user_id=row["userId"],
        movie_id=row["tmdbId"],
        train_data=train
    )

    predictions.append(predicted)


evaluation_rmse_mae = evaluation_8000.copy()

evaluation_rmse_mae["predicted_rating"] = predictions

# Remove records where prediction could not be generated
evaluation_rmse_mae = (
    evaluation_rmse_mae
    .dropna(subset=["predicted_rating"])
)


# RMSE
rmse = np.sqrt(
    mean_squared_error(
        evaluation_rmse_mae["rating"],
        evaluation_rmse_mae["predicted_rating"]
    )
)


# MAE
mae = mean_absolute_error(
    evaluation_rmse_mae["rating"],
    evaluation_rmse_mae["predicted_rating"]
)


print("\nContent-Based Filtering - Rating Prediction")
print("---------------------------------------------")
print(f"Valid predictions: {len(evaluation_rmse_mae):,}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")


# ============================================================
# 10. TOP-K RECOMMENDATION FUNCTION
# ============================================================

def get_top_k_recommendations(
    user_id,
    train_data,
    k=10
):

    user_history = train_data[
        train_data["userId"] == user_id
    ]

    if user_history.empty:
        return []

    # Movies already rated by user
    rated_movies = set(
        user_history["tmdbId"]
    )

    # Use highly-rated movies as preference seeds
    liked_movies = user_history[
        user_history["rating"] >= RELEVANCE_THRESHOLD
    ]

    if liked_movies.empty:
        return []

    scores = {}

    for _, row in liked_movies.iterrows():

        movie_id = row["tmdbId"]

        target_index = movies.index[
            movies["tmdbId"] == movie_id
        ]

        if len(target_index) == 0:
            continue

        target_index = target_index[0]

        similarities = cosine_sim[target_index]

        for movie_index, similarity in enumerate(
            similarities
        ):

            candidate_id = movies.iloc[
                movie_index
            ]["tmdbId"]

            # Do not recommend movies already rated
            if candidate_id in rated_movies:
                continue

            if candidate_id == movie_id:
                continue

            # Weighted recommendation score
            weighted_score = (
                similarity * row["rating"]
            )

            if candidate_id not in scores:
                scores[candidate_id] = 0

            scores[candidate_id] += weighted_score

    # Rank movies
    ranked_movies = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        movie_id
        for movie_id, score in ranked_movies[:k]
    ]


# ============================================================
# 11. TOP-K EVALUATION
#    Uses the SAME topk_users, K and threshold
# ============================================================

topk_results = []

for user_id in topk_users:

    # Generate Top-K recommendations
    recommended = get_top_k_recommendations(
        user_id,
        train,
        K
    )

    if len(recommended) == 0:
        continue

    # Get user's relevant movies from TEST set
    user_test = test[
        test["userId"] == user_id
    ]

    relevant_movies = set(
        user_test[
            user_test["rating"] >= RELEVANCE_THRESHOLD
        ]["tmdbId"]
    )

    if len(relevant_movies) == 0:
        continue

    recommended_set = set(recommended)

    # Number of relevant recommendations
    hits = len(
        recommended_set.intersection(
            relevant_movies
        )
    )

    # Precision@K
    precision = hits / K

    # Recall@K
    recall = hits / len(relevant_movies)

    # F1@K
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    topk_results.append({
        "userId": user_id,
        "precision@10": precision,
        "recall@10": recall,
        "f1@10": f1
    })


topk_results_df = pd.DataFrame(
    topk_results
)


# ============================================================
# 12. TOP-K FINAL RESULTS
# ============================================================

mean_precision = (
    topk_results_df["precision@10"].mean()
)

mean_recall = (
    topk_results_df["recall@10"].mean()
)

mean_f1 = (
    topk_results_df["f1@10"].mean()
)


print("\nContent-Based Filtering - Top-K Results")
print("----------------------------------------")
print(
    f"Users evaluated: {len(topk_results_df)}"
)
print(
    f"Precision@10: {mean_precision:.4f}"
)
print(
    f"Recall@10:    {mean_recall:.4f}"
)
print(
    f"F1@10:        {mean_f1:.4f}"
)


# ============================================================
# 13. FINAL RESULTS TABLE
# ============================================================

content_based_results = pd.DataFrame({
    "Metric": [
        "RMSE",
        "MAE",
        "Precision@10",
        "Recall@10",
        "F1@10"
    ],
    "Content-Based Filtering": [
        rmse,
        mae,
        mean_precision,
        mean_recall,
        mean_f1
    ]
})


print("\n========================================")
print("FINAL CONTENT-BASED FILTERING RESULTS")
print("========================================")

print(
    content_based_results.to_string(
        index=False
    )
)


# ============================================================
# 14. EXAMPLE RECOMMENDATION
# ============================================================

example_movie = "The Shawshank Redemption"

recommendations = recommend_movies(
    example_movie,
    K
)

print(
    f"\nTop {K} recommendations for: "
    f"{example_movie}"
)

print(
    recommendations.to_string(
        index=False
    )
)