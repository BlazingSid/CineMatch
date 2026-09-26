from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


def prepare_surprise_data(ratings):
    """Convert ratings DataFrame into Surprise format."""

    reader = Reader(
        rating_scale=(
            ratings["rating"].min(),
            ratings["rating"].max()
        )
    )

    data = Dataset.load_from_df(
        ratings[
            ["userId", "movieId", "rating"]
        ],
        reader
    )

    return data


def train_svd(
    ratings,
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    test_size=0.2,
    random_state=42
):
    """
    Train an SVD collaborative filtering model.

    Returns:
        model
        testset
        rmse
    """

    data = prepare_surprise_data(ratings)

    trainset, testset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state
    )

    model.fit(trainset)

    predictions = model.test(testset)

    rmse = accuracy.rmse(
        predictions,
        verbose=False
    )

    return model, testset, rmse