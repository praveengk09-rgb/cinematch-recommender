"""Spark-free recommender built from exported ALS factors (used by API + Streamlit)."""
import numpy as np
import pandas as pd

import config
from core import ItemSpace, MovieNotFoundError, UserNotFoundError


class LocalRecommender:
    def __init__(self, factors_path=config.FACTORS_PATH,
                 ratings_csv=config.RATINGS_CSV, movies_csv=config.MOVIES_CSV):
        if not factors_path.exists():
            raise FileNotFoundError(
                f"{factors_path} missing. Train first: SAVE_MODEL=1 python train.py")
        z = np.load(factors_path)
        self.U = z["user_factors"].astype(np.float64)
        self.user_pos = {int(u): i for i, u in enumerate(z["user_ids"])}
        movies = pd.read_csv(movies_csv)
        self.space = ItemSpace(z["item_ids"], z["item_factors"], movies)
        ratings = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "rating"])
        self.ratings = ratings
        self.seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()

    @property
    def user_ids(self):
        return sorted(self.user_pos)

    @property
    def titles(self):
        m = self.space.movies
        return sorted(m.loc[m.index.isin(self.space.pos), "title"].dropna())

    def recommend_top_n_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if n < 1:
            raise ValueError("n must be >= 1")
        if int(user_id) not in self.user_pos:
            raise UserNotFoundError(f"userId {user_id} not found in the trained model.")
        scores = self.space.V @ self.U[self.user_pos[int(user_id)]]
        seen = self.seen.get(int(user_id), set())
        if seen:  # never re-recommend movies the user already rated
            scores = np.where(np.isin(self.space.ids, list(seen)), -np.inf, scores)
        order = [i for i in np.argsort(-scores)[:n] if np.isfinite(scores[i])]
        out = self.space.describe([int(self.space.ids[i]) for i in order])
        out["predicted_rating"] = np.round(
            np.clip(scores[order], config.RATING_MIN, config.RATING_MAX), 3)
        return out

    def get_similar_movies(self, movie_title: str, top_k: int = 5) -> pd.DataFrame:
        return self.space.similar(self.space.resolve_title(movie_title), top_k)

    def get_similar_by_id(self, movie_id: int, top_k: int = 5) -> pd.DataFrame:
        return self.space.similar(movie_id, top_k)

    def user_history(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if int(user_id) not in self.user_pos:
            raise UserNotFoundError(f"userId {user_id} not found.")
        r = self.ratings[self.ratings.userId == int(user_id)].nlargest(n, "rating")
        out = self.space.describe(r["movieId"].tolist())
        out["rating"] = r["rating"].to_numpy()
        return out
