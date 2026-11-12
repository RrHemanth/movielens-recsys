import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recsys.serving.app import create_app


@pytest.fixture
def client(tmp_path):
    # Movies 1-3: animated (Toy Story, Up, Nemo), 4: Monsters Inc., 5: Heat. 20-23: popular list.
    pd.DataFrame({"user_id": [7, 7, 7], "rank": [1, 2, 3], "movie_id": [5, 4, 1]}).to_parquet(
        tmp_path / "recommendations.parquet")
    pd.DataFrame({"movie_id": [20, 21, 22, 23]}).to_parquet(tmp_path / "popular.parquet")
    ids = [1, 2, 3, 4, 5, 20, 21, 22, 23]
    pd.DataFrame({"movie_id": ids, "title": [f"Movie {i}" for i in ids], "genres": "x"}).to_parquet(
        tmp_path / "movies.parquet")
    vectors = [[0.9, 0.1], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1], [0.1, 0.9]]
    pd.DataFrame({"movie_id": [1, 2, 3, 4, 5], "vector": vectors}).to_parquet(tmp_path / "movie_vectors.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"as_of": "2019-01-01", "n_users_precomputed": 1}))
    return TestClient(create_app(tmp_path))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["as_of"] == "2019-01-01"


def test_known_user_gets_their_precomputed_list(client):
    body = client.get("/recommendations/7").json()
    assert body["source"] == "re-ranker"
    assert [m["movie_id"] for m in body["movies"]] == [5, 4, 1]
    assert body["movies"][0]["title"] == "Movie 5"


def test_unknown_user_gets_popularity(client):
    body = client.get("/recommendations/999?k=2").json()
    assert body["source"] == "popularity"
    assert [m["movie_id"] for m in body["movies"]] == [20, 21]


def test_new_user_who_liked_animation_gets_animation(client):
    body = client.post("/recommendations/new-user", json={"liked_movie_ids": [1, 2, 3], "k": 1}).json()
    assert body["source"] == "fold-in"
    assert [m["movie_id"] for m in body["movies"]] == [4]   # Monsters Inc., not Heat


def test_new_user_with_only_unknown_movies_gets_popularity_minus_their_movies(client):
    body = client.post("/recommendations/new-user", json={"liked_movie_ids": [20, 999], "k": 2}).json()
    assert body["source"] == "popularity"
    assert [m["movie_id"] for m in body["movies"]] == [21, 22]   # 20 skipped: they already liked it


@pytest.mark.parametrize("request_args", [
    ("get", "/recommendations/abc", None),                                  # user id isn't a number
    ("get", "/recommendations/7?k=50", None),                               # k too large
    ("post", "/recommendations/new-user", {"liked_movie_ids": []}),         # nothing liked
])
def test_bad_input_is_rejected_with_422(client, request_args):
    method, url, body = request_args
    response = client.get(url) if method == "get" else client.post(url, json=body)
    assert response.status_code == 422