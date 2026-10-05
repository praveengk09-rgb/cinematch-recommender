"""Spark-native inference engine (Functions 1 & 2 from the spec)."""
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

import config
from core import ItemSpace, MovieNotFoundError, UserNotFoundError


class SparkRecommender:
    def __init__(self, spark: SparkSession, model: ALSModel,
                 ratings: DataFrame, movies: DataFrame):
        self.spark, self.model, self.ratings, self.movies = spark, model, ratings, movies
        # Latent item matrix V extracted from model.itemFactors (collected once, ~9K x rank).
        items = model.itemFactors.toPandas()
        self.space = ItemSpace(items["id"].to_numpy(), items["features"].to_list(),
                               movies.toPandas())

    def recommend_top_n_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Top-N unseen movies for a user, with titles/genres and clipped ratings."""
        if n < 1:
            raise ValueError("n must be >= 1")
        uid = int(user_id)
        if self.model.userFactors.filter(F.col("id") == uid).count() == 0:
            raise UserNotFoundError(f"userId {uid} not found in the trained model.")
        seen = self.ratings.filter(F.col("userId") == uid).select("movieId")
        users = self.spark.createDataFrame([(uid,)], ["userId"])
        recs = (self.model.recommendForUserSubset(users, n + seen.count())  # over-fetch, then drop seen
                .select(F.explode("recommendations").alias("r"))
                .select(F.col("r.movieId").alias("movieId"), F.col("r.rating").alias("raw")))
        recs = (recs.join(seen, "movieId", "left_anti")
                .orderBy(F.desc("raw")).limit(n)                     # rank on the raw score
                .join(self.movies, "movieId")
                .withColumn("predicted_rating", F.round(F.least(F.greatest(
                    F.col("raw"), F.lit(config.RATING_MIN)), F.lit(config.RATING_MAX)), 3))
                .orderBy(F.desc("raw"))
                .select("movieId", "title", "genres", "predicted_rating"))
        return recs.toPandas()

    def get_similar_movies(self, movie_title: str, top_k: int = 5) -> pd.DataFrame:
        """Exact cosine similarity over latent item vectors V."""
        return self.space.similar(self.space.resolve_title(movie_title), top_k)


# --- Module-level wrappers matching the requested function signatures ---------------------
_ENGINE: SparkRecommender | None = None


def set_engine(engine: SparkRecommender) -> None:
    global _ENGINE
    _ENGINE = engine


def _engine() -> SparkRecommender:
    if _ENGINE is None:
        raise RuntimeError("Call set_engine(SparkRecommender(...)) first.")
    return _ENGINE


def recommend_top_n_for_user(user_id: int, n: int = 10) -> pd.DataFrame:
    return _engine().recommend_top_n_for_user(user_id, n)


def get_similar_movies(movie_title: str, top_k: int = 5) -> pd.DataFrame:
    return _engine().get_similar_movies(movie_title, top_k)
