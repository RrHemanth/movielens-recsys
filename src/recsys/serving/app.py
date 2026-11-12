import json
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from recsys.config import load_config
from recsys.models.cold_start import FoldInRecommender, top_unseen
from recsys.models.similarity import load_vectors


# ---------- What requests and responses look like ----------
class Movie(BaseModel):
    movie_id: int
    title: str


class RecommendationResponse(BaseModel):
    user_id: int | None
    source: Literal["re-ranker", "fold-in", "popularity"]   # which model produced the list
    movies: list[Movie]


class NewUserRequest(BaseModel):
    liked_movie_ids: list[int] = Field(min_length=1, max_length=100)
    k: int = Field(10, ge=1, le=10)


# ---------- Loads the batch job's files once, answers requests from memory ----------
class RecommendationStore:
    def __init__(self, artifacts_dir: Path):
        recs = pd.read_parquet(artifacts_dir / "recommendations.parquet").sort_values(["user_id", "rank"])
        self.precomputed = recs.groupby("user_id")["movie_id"].apply(list).to_dict()
        self.popular = pd.read_parquet(artifacts_dir / "popular.parquet")["movie_id"].tolist()
        movies = pd.read_parquet(artifacts_dir / "movies.parquet")
        self.titles = {int(m): t for m, t in zip(movies["movie_id"], movies["title"])}
        movie_ids, vectors = load_vectors(artifacts_dir / "movie_vectors.parquet")
        self.fold_in = FoldInRecommender(movie_ids, vectors)
        self.manifest = json.loads((artifacts_dir / "manifest.json").read_text())

    def for_user(self, user_id: int, k: int) -> tuple[str, list]:
        if user_id in self.precomputed:
            return "re-ranker", self.precomputed[user_id][:k]
        return "popularity", self.popular[:k]

    def for_new_user(self, liked: list, k: int) -> tuple[str, list]:
        onboard = pd.DataFrame({"user_id": 0, "movie_id": liked, "rating": 5.0})   # liked = rated 5
        recs = self.fold_in.recommend(onboard, k=k)
        if 0 in recs:
            return "fold-in", recs[0]
        return "popularity", top_unseen(self.popular, set(liked), k)   # none of the liked movies are known


# ---------- The web app ----------
def create_app(artifacts_dir=None) -> FastAPI:
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else load_config()["paths"]["serving"]
    store = RecommendationStore(artifacts_dir)
    app = FastAPI(title="MovieLens recommender", version="1.0")

    def to_movies(ids) -> list[Movie]:
        return [Movie(movie_id=int(m), title=store.titles.get(int(m), "unknown")) for m in ids]

    @app.get("/health")
    def health():
        keys = ["as_of", "n_users_precomputed", "git_commit"]
        return {"status": "ok", **{key: store.manifest.get(key) for key in keys}}

    @app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
    def recommend_for_user(user_id: int, k: int = Query(10, ge=1, le=10)):
        source, ids = store.for_user(user_id, k)
        return RecommendationResponse(user_id=user_id, source=source, movies=to_movies(ids))

    @app.post("/recommendations/new-user", response_model=RecommendationResponse)
    def recommend_for_new_user(request: NewUserRequest):
        source, ids = store.for_new_user(request.liked_movie_ids, request.k)
        return RecommendationResponse(user_id=None, source=source, movies=to_movies(ids))

    return app