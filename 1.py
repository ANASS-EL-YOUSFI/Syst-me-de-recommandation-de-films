import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

print(ratings.head())
print(movies.head())


print("\nStatistiques descriptives :")
print(ratings.describe())


print("Utilisateurs :", ratings["userId"].nunique())
print("Films :", ratings["movieId"].nunique())


plt.hist(ratings["rating"], bins=10)
plt.xlabel("Note")
plt.ylabel("Fréquence")
plt.title("Distribution des notes")
plt.show()



ratings = ratings.dropna()
movies = movies.dropna()

ratings = ratings.drop_duplicates()
movies = movies.drop_duplicates()


data = pd.merge(ratings, movies, on="movieId")


user_counts = data["userId"].value_counts()
active_users = user_counts[user_counts >= 20].index
data = data[data["userId"].isin(active_users)]


movie_counts = data["movieId"].value_counts()
popular_movies = movie_counts[movie_counts >= 20].index
data = data[data["movieId"].isin(popular_movies)]

print("Données après filtrage :", data.shape)



# MODÉLISATION

# ---------- 1. Recommandation par popularité ----------
def RecommandationParPopularite(df, top_n=10, min_ratings=4):
    popularity = (
        df.groupby('title')
        .agg(
            mean_rating=('rating', 'mean'),
            rating_count=('rating', 'count'),
        )
        .reset_index()
    )


    popularity = popularity[popularity['rating_count'] >= min_ratings]


    popularity = popularity.sort_values(
        by=['rating_count', 'mean_rating'],
        ascending=False
    )

    return popularity.head(top_n)

# ---------- 2. Filtrage collaboratif (Item-Based) ----------
user_movie_matrix = data.pivot_table(
    index="userId",
    columns="title",
    values="rating"
).fillna(0)

item_similarity = cosine_similarity(user_movie_matrix.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

def recommend_item_based(user_id, n=5):
    user_ratings = user_movie_matrix.loc[user_id]
    liked_movies = user_ratings[user_ratings >= 4].index

    scores = item_similarity_df[liked_movies].sum(axis=1)
    scores = scores.drop(liked_movies)

    return scores.sort_values(ascending=False).head(n)


# ---------- 3. Option avancée : SVD (sklearn) ----------
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_movie_matrix)

predicted_ratings = np.dot(latent_matrix, svd.components_)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)



# . SÉPARATION TRAIN / TEST

train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)



#  ÉVALUATION DU MODÈLE (RMSE / MAE)


# 1️ Évaluation Recommandation par Popularité
print("\n=== Évaluation Recommandation par Popularité ===")
top_popular = RecommandationParPopularite(data, top_n=10)
print(top_popular[['title', 'mean_rating', 'rating_count']])

# 2️ Évaluation Filtrage collaboratif Item-Based (corrigé)
print("\n=== Évaluation Filtrage collaboratif (Item-Based) ===")
y_true_item = []
y_pred_item = []

for _, row in test_data.iterrows():
    user = row["userId"]
    movie = row["title"]

    if movie in item_similarity_df.columns and user in user_movie_matrix.index:
        user_ratings = user_movie_matrix.loc[user]
        liked_movies = user_ratings[user_ratings >= 4].index

        if len(liked_movies) > 0:

            sim_scores = item_similarity_df.loc[movie, liked_movies]
            ratings = user_ratings[liked_movies]
            if sim_scores.sum() > 0:
                pred = np.dot(sim_scores, ratings) / sim_scores.sum()
            else:
                pred = ratings.mean()

            y_pred_item.append(pred)
            y_true_item.append(row["rating"])


if len(y_pred_item) > 0:
    rmse_item = np.sqrt(mean_squared_error(y_true_item, y_pred_item))
    mae_item = mean_absolute_error(y_true_item, y_pred_item)
    print("RMSE Item-Based :", rmse_item)
    print("MAE Item-Based  :", mae_item)
else:
    print("Pas assez de données pour calculer RMSE/MAE Item-Based")


# 3️ Évaluation SVD
print("\n=== Évaluation SVD ===")
y_true_svd = []
y_pred_svd = []

for _, row in test_data.iterrows():
    user = row["userId"]
    movie = row["title"]
    if user in predicted_ratings_df.index and movie in predicted_ratings_df.columns:
        y_true_svd.append(row["rating"])
        y_pred_svd.append(predicted_ratings_df.loc[user, movie])

rmse_svd = np.sqrt(mean_squared_error(y_true_svd, y_pred_svd))
mae_svd = mean_absolute_error(y_true_svd, y_pred_svd)

print("RMSE SVD :", rmse_svd)
print("MAE SVD  :", mae_svd)

#  RECOMMANDATION TOP-N


def recommend_svd(user_id, n=5):
    user_predictions = predicted_ratings_df.loc[user_id]
    already_seen = user_movie_matrix.loc[user_id]
    user_predictions = user_predictions[already_seen == 0]
    return user_predictions.sort_values(ascending=False).head(n)

print("\nRecommandation Item-Based (User 1) :")
print(recommend_item_based(user_id=1, n=5))

print("\nRecommandation SVD (User 1) :")
print(recommend_svd(user_id=1, n=5))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
        algo = request.args.get('algo')

        if algo == "item":
            recs = recommend_item_based(user_id).index.tolist()
        else:
            recs = recommend_svd(user_id).index.tolist()

        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
