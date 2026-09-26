
# CineMatch — Evaluation Report

## Objective

CineMatch compares three collaborative filtering approaches:

1. User-Based Collaborative Filtering
2. Item-Based Collaborative Filtering
3. Singular Value Decomposition (SVD)

The models are evaluated using Root Mean Squared Error (RMSE).

## Evaluation Setup

The dataset was divided into:

- 80% training data
- 20% testing data
- Random state: 42

The same train/test split was used for all three approaches.

## RMSE Results

| Model | RMSE |
|---|---:|
| SVD | 0.8831 |
| User-Based CF | 0.9684 |
| Item-Based CF | 1.1946 |


## Result

The model with the lowest RMSE in this experiment was:

**SVD**

with an RMSE of **0.8831**.

## Interpretation

RMSE measures the difference between predicted and actual ratings.
A lower RMSE indicates smaller prediction error on the evaluated test set.

## Conclusion

CineMatch demonstrates three approaches to collaborative filtering:

- User-User similarity
- Item-Item similarity
- Matrix Factorization using SVD

The comparison provides a quantitative view of their rating-prediction performance on the MovieLens dataset.
