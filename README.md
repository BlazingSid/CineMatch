# 🎬 CineMatch

### Movie Recommendation System using Collaborative Filtering

CineMatch is a movie recommendation system built using the **MovieLens dataset** and collaborative filtering techniques.

The project explores two major approaches:

* **Memory-Based Collaborative Filtering**

  * User-User Collaborative Filtering
  * Item-Item Collaborative Filtering
  * Cosine Similarity
* **Model-Based Collaborative Filtering**

  * Matrix Factorization
  * Singular Value Decomposition (SVD)

The system learns from historical user ratings and generates personalized movie recommendations.

---

## 🚀 Features

* MovieLens dataset analysis
* Rating distribution analysis
* User activity analysis
* Utility matrix construction
* Manual cosine similarity implementation using NumPy
* User-User collaborative filtering
* Item-Item collaborative filtering
* SVD-based matrix factorization
* RMSE-based model evaluation
* Top-N personalized recommendations
* Recommendation result visualizations
* Reusable Python modules in `src/`
* Complete experimentation notebook

---

## 🧠 How CineMatch Works

The recommendation pipeline can be summarized as:

```text
MovieLens Ratings
       │
       ▼
Data Exploration & Cleaning
       │
       ▼
User × Movie Utility Matrix
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
Memory-Based CF         Model-Based CF
       │                      │
   ┌───┴────┐                 │
   ▼        ▼                 ▼
User-User  Item-Item          SVD
   │        │                 │
   └───┬────┘                 │
       │                      │
       └──────────┬───────────┘
                  ▼
          Predicted Preferences
                  │
                  ▼
           Top-N Recommendations
```

---

## 🔍 Collaborative Filtering Approaches

### 1. User-User Collaborative Filtering

User-User collaborative filtering finds users with similar rating patterns.

For a target user:

```text
Target User
     │
     ▼
Find similar users
     │
     ▼
Look at movies they liked
     │
     ▼
Remove movies already watched
     │
     ▼
Rank candidates
     │
     ▼
Top-N recommendations
```

Similarity is calculated using **cosine similarity**.

---

### 2. Item-Item Collaborative Filtering

Item-Item collaborative filtering compares movies based on how users have rated them.

```text
User's liked movies
        │
        ▼
Find similar movies
        │
        ▼
Remove already-rated movies
        │
        ▼
Rank candidates
        │
        ▼
Top-N recommendations
```

This approach creates a movie-to-movie similarity representation from the user-rating matrix.

---

### 3. SVD / Matrix Factorization

CineMatch also uses **Singular Value Decomposition (SVD)** through the Surprise library.

The model represents users and movies using latent factors and learns their interactions to predict ratings.

```text
User Ratings
     │
     ▼
Matrix Factorization
     │
     ├── User Factors
     │
     └── Movie Factors
            │
            ▼
      Predicted Ratings
            │
            ▼
       Top-N Movies
```

---

## 📊 Dataset

CineMatch uses the **MovieLens dataset**.

The main datasets used are:

### `movies.csv`

Contains:

* `movieId`
* `title`
* `genres`

### `ratings.csv`

Contains:

* `userId`
* `movieId`
* `rating`
* `timestamp`

The rating matrix is highly sparse, which makes collaborative filtering an appropriate problem to explore.

---

## 📈 Exploratory Data Analysis

The project includes visualizations for:

### Rating Distribution

Shows how ratings are distributed across the dataset.

### Most-Rated Movies

Identifies movies with the highest number of ratings.

### Ratings per User

Shows the distribution of user activity.

These visualizations help understand the structure and sparsity of the recommendation dataset.

---

## 🤖 Model Evaluation

The SVD model was evaluated using **Root Mean Squared Error (RMSE)**.

Current SVD result:

```text
SVD RMSE: 0.8807
```

The notebook also contains recommendation experiments and evaluation work for the collaborative filtering approaches.

> RMSE is used to measure the difference between predicted ratings and actual ratings. Lower values indicate smaller prediction error on the evaluated test set.

---

## 🎯 Recommendation Example

CineMatch can generate personalized Top-N recommendations for a user.

Example workflow:

```python
recommendations = recommend_movies_svd(
    model=model,
    user_id=1,
    ratings=ratings,
    movies=movies,
    n=5
)
```

The resulting recommendations contain:

```text
movieId
title
genres
predicted_rating
```

---

## 📊 Project Visualizations

Generated plots are stored in:

```text
outputs/plots/
```

Current visualizations include:

```text
rating_distribution.png
top_10_most_rated_movies.png
ratings_per_user.png
svd_rmse.png
top_5_recommendations.png
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Scikit-Surprise

### Visualization

* Matplotlib
* Seaborn

### Development

* Jupyter Notebook
* VS Code
* Git / GitHub

---

## 📁 Project Structure

```text
CineMatch/
│
├── data/
│   ├── movies.csv
│   └── ratings.csv
│
├── notebooks/
│   └── cinematch.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── similarity.py
│   ├── model.py
│   └── recommender.py
│
├── outputs/
│   ├── plots/
│   └── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CineMatch
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run the notebook

Open:

```text
notebooks/cinematch.ipynb
```

and run the cells sequentially.

### Run the Python implementation

The reusable implementation is located inside:

```text
src/
```

The modules provide functionality for:

* Loading MovieLens data
* Building the utility matrix
* Calculating similarity
* Training the SVD model
* Generating recommendations

---

## 🧪 Project Verification

The reusable implementation was tested independently from the notebook using a project-level smoke test.

The test verifies:

```text
✓ Dataset loading
✓ Utility matrix creation
✓ SVD training
✓ RMSE calculation
✓ Movie recommendation generation
```

---

## 🔮 Future Improvements

Possible extensions include:

* Precision@K evaluation
* Recall@K evaluation
* Hybrid recommendation models
* Popularity-based cold-start recommendations
* Genre-aware recommendations
* Better handling of sparse user histories
* Hyperparameter tuning
* Interactive recommendation interface
* API deployment using FastAPI

---

## 📚 Project Goals

CineMatch was built to understand the complete recommendation-system pipeline:

```text
Data
 ↓
Exploration
 ↓
Utility Matrix
 ↓
Similarity
 ↓
Collaborative Filtering
 ↓
Matrix Factorization
 ↓
Evaluation
 ↓
Personalized Recommendations
```

The main goal is not only to generate recommendations, but to understand the mathematical and engineering concepts behind collaborative filtering systems.

---

## 👨‍💻 Project

**CineMatch — Movie Recommendation System**

Built with Python, Pandas, NumPy, Scikit-learn, Scikit-Surprise, Matplotlib and Seaborn.
