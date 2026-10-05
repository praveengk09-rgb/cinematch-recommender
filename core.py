"""Spark-free building blocks shared by the training demo, REST API and Streamlit UI."""
import difflib
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class UserNotFoundError(LookupError):
    pass


class MovieNotFoundError(LookupError):
    pass


class ItemSpace:
    """Item latent matrix V (n_items x rank) + exact cosine-similarity search."""

    def __init__(self, item_ids, factors, movies: pd.DataFrame):
        self.ids = np.asarray(item_ids, dtype=np.int64)
        self.V = np.asarray(factors, dtype=np.float64)
        self.movies = movies.drop_duplicates("movieId").set_index("movieId")
        self.pos = {int(m): i for i, m in enumerate(self.ids)}

    def describe(self, movie_ids) -> pd.DataFrame:
        """Attach title/genres for a list of movieIds."""
        meta = self.movies.reindex(movie_ids)[["title", "genres"]]
        return pd.DataFrame({"movieId": list(movie_ids),
                             "title": meta["title"].to_numpy(),
                             "genres": meta["genres"].to_numpy()})

    def resolve_title(self, title: str) -> int:
        """Exact (case-insensitive) match first, then substring; else raise with suggestions."""
        q = (title or "").strip().lower()
        if not q:
            raise MovieNotFoundError("Movie title is empty.")
        lowered = self.movies["title"].str.lower()
        for hits in (self.movies[lowered == q],
                     self.movies[lowered.str.contains(re.escape(q), na=False)]):
            hits = hits[hits.index.isin(self.pos)]       # must have a latent vector
            if len(hits):
                return int(hits.sort_index().index[0])
        close = difflib.get_close_matches(title, self.movies["title"].dropna().tolist(), n=3)
        hint = f" Did you mean: {close}?" if close else ""
        raise MovieNotFoundError(f"No rated movie matches title '{title}'.{hint}")

    def similar(self, movie_id: int, top_k: int = 5) -> pd.DataFrame:
        """Exact cosine similarity of one item vector against the whole V matrix."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if int(movie_id) not in self.pos:
            raise MovieNotFoundError(f"movieId {movie_id} has no latent factors (never rated?).")
        q = self.V[self.pos[int(movie_id)]][None, :]
        sims = np.nan_to_num(1.0 - cdist(q, self.V, metric="cosine")[0])  # cosine = 1 - distance
        order = [i for i in np.argsort(-sims) if self.ids[i] != movie_id][:top_k]
        out = self.describe([int(self.ids[i]) for i in order])
        out["similarity"] = np.round(sims[order], 4)
        return out
