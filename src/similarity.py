import numpy as np
import pandas as pd


def cosine_similarity_manual(a, b):
    """
    Calculate cosine similarity using only NumPy.

    Missing values are ignored by comparing
    only commonly rated items.
    """

    mask = ~np.isnan(a) & ~np.isnan(b)

    a_common = a[mask]
    b_common = b[mask]

    if len(a_common) == 0:
        return 0.0

    denominator = (
        np.linalg.norm(a_common)
        * np.linalg.norm(b_common)
    )

    if denominator == 0:
        return 0.0

    return np.dot(
        a_common,
        b_common
    ) / denominator


def build_utility_matrix(ratings):
    """
    Create a user-item utility matrix.

    Rows    → users
    Columns → movies
    Values  → ratings
    """

    return ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    )


def build_item_matrix(utility_matrix):
    """
    Transpose the utility matrix.

    Rows    → movies
    Columns → users
    """

    return utility_matrix.T