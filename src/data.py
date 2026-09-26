import pandas as pd


def load_movies(path="data/movies.csv"):
    """Load MovieLens movie metadata."""
    return pd.read_csv(path)


def load_ratings(path="data/ratings.csv"):
    """Load MovieLens ratings."""
    return pd.read_csv(path)


def load_data(
    movies_path="data/movies.csv",
    ratings_path="data/ratings.csv"
):
    """Load movies and ratings datasets."""
    movies = load_movies(movies_path)
    ratings = load_ratings(ratings_path)

    return movies, ratings