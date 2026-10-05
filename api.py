"""REST API.  Run:  uvicorn api:app --reload --port 8000   (docs at /docs)"""
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from core import MovieNotFoundError, UserNotFoundError
from serving import LocalRecommender

app = FastAPI(title="Movie Recommender (ALS)", version="1.0")


@lru_cache(maxsize=1)
def engine() -> LocalRecommender:
    try:
        return LocalRecommender()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(user_id: int = Query(..., ge=1), top_n: int = Query(10, ge=1, le=100)):
    try:
        recs = engine().recommend_top_n_for_user(user_id, top_n)
    except UserNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return {"user_id": user_id, "recommendations": recs.to_dict("records")}


@app.get("/similar")
def similar(movie_id: Optional[int] = Query(None, ge=1),
            title: Optional[str] = Query(None, description="Alternative to movie_id"),
            top_k: int = Query(5, ge=1, le=50)):
    if movie_id is None and not title:
        raise HTTPException(422, "Provide movie_id (or title).")
    try:
        eng = engine()
        res = eng.get_similar_by_id(movie_id, top_k) if movie_id is not None \
            else eng.get_similar_movies(title, top_k)
    except MovieNotFoundError as exc:
        raise HTTPException(404, str(exc))
    return {"query": movie_id if movie_id is not None else title, "similar": res.to_dict("records")}
