import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


def recommend_movies_svd(
    model,
    user_id,
    ratings,
    movies,
    n=5
):
    """
    Generate top-N movie recommendations
    using a trained SVD model.
    """

    rated_movie_ids = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    unrated_movies = movies[
        ~movies["movieId"].isin(
            rated_movie_ids
        )
    ]

    predictions = []

    for movie_id in unrated_movies["movieId"]:

        prediction = model.predict(
            user_id,
            movie_id
        )

        predictions.append({
            "movieId": movie_id,
            "predicted_rating": prediction.est
        })

    recommendations = (
        pd.DataFrame(predictions)
        .sort_values(
            "predicted_rating",
            ascending=False
        )
        .head(n)
    )

    return recommendations.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId"
    )


def recommend_user_based(
    user_id,
    ratings,
    utility_matrix,
    movies,
    n=5,
    neighbor_count=20
):
    """
    Generate recommendations using
    user-user collaborative filtering.
    """

    rated_movies = set(
        ratings[
            ratings["userId"] == user_id
        ]["movieId"]
    )

    target_vector = (
        utility_matrix
        .loc[user_id]
        .values
    )

    similarities = {}

    for other_user in utility_matrix.index:

        if other_user == user_id:
            continue

        other_vector = (
            utility_matrix
            .loc[other_user]
            .values
        )

        mask = (
            ~np.isnan(target_vector)
            & ~np.isnan(other_vector)
        )

        if not mask.any():
            similarity = 0.0
        else:
            a = target_vector[mask]
            b = other_vector[mask]

            denominator = (
                np.linalg.norm(a)
                * np.linalg.norm(b)
            )

            similarity = (
                np.dot(a, b) / denominator
                if denominator != 0
                else 0.0
            )

        similarities[other_user] = similarity

    top_users = (
        pd.Series(similarities)
        .sort_values(ascending=False)
        .head(neighbor_count)
    )

    candidate_scores = {}

    for other_user, similarity in top_users.items():

        liked_movies = ratings[
            (ratings["userId"] == other_user)
            & (ratings["rating"] >= 4)
        ]

        for _, row in liked_movies.iterrows():

            movie_id = row["movieId"]

            if movie_id in rated_movies:
                continue

            candidate_scores[movie_id] = (
                candidate_scores.get(movie_id, 0)
                + similarity * row["rating"]
            )

    recommendations = (
        pd.Series(
            candidate_scores,
            name="score"
        )
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    recommendations = recommendations.rename(
        columns={"index": "movieId"}
    )

    return recommendations.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId"
    )


def recommend_item_based(
    user_id,
    ratings,
    movies,
    item_matrix,
    item_similarity_matrix,
    n=5,
    similar_per_movie=10
):
    """
    Generate recommendations using
    item-item collaborative filtering.
    """

    user_ratings = ratings[
        ratings["userId"] == user_id
    ]

    already_rated = set(
        user_ratings["movieId"]
    )

    liked_movies = user_ratings[
        user_ratings["rating"] >= 4
    ]

    candidate_scores = {}

    for movie_id in liked_movies["movieId"]:

        if movie_id not in item_matrix.index:
            continue

        movie_index = (
            item_matrix.index
            .get_loc(movie_id)
        )

        similarities = (
            item_similarity_matrix[movie_index]
        )

        sorted_indices = np.argsort(
            similarities
        )[::-1]

        count = 0

        for index in sorted_indices:

            candidate_id = (
                item_matrix.index[index]
            )

            if candidate_id == movie_id:
                continue

            if candidate_id in already_rated:
                continue

            candidate_scores[candidate_id] = (
                candidate_scores.get(candidate_id, 0)
                + similarities[index]
            )

            count += 1

            if count >= similar_per_movie:
                break

    recommendations = (
        pd.Series(
            candidate_scores,
            name="score"
        )
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    recommendations = recommendations.rename(
        columns={"index": "movieId"}
    )

    return recommendations.merge(
        movies[
            ["movieId", "title", "genres"]
        ],
        on="movieId"
    )