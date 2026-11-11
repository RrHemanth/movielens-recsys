import numpy as np
import pandas as pd
import pytest

from recsys.features import add_features, add_labels, genre_match, movie_year

GENRES = {1: {"Sci-Fi"}, 2: {"Sci-Fi", "Action"}, 3: {"Comedy"},
          4: {"Sci-Fi", "Action"}, 5: {"Drama"}}


def test_movie_year_from_title():
    assert movie_year("Inception (2010)") == 2010
    assert np.isnan(movie_year("Some Title Without a Year"))


def test_genre_match_example():
    # User 1 liked movies 1, 2, 3 -> Sci-Fi in 2/3 of likes, Action 1/3, Comedy 1/3.
    past_likes = pd.DataFrame({"user_id": [1, 1, 1], "movie_id": [1, 2, 3]})
    cands = pd.DataFrame({"user_id": [1, 1], "movie_id": [4, 5]})
    scores = genre_match(cands, past_likes, GENRES)
    assert scores[0] == pytest.approx((2 / 3 + 1 / 3) / 2)   # movie 4: Sci-Fi + Action
    assert scores[1] == pytest.approx(0.0)                   # movie 5: Drama, never liked


def test_labels():
    cands = pd.DataFrame({"user_id": [1, 1], "movie_id": [10, 11]})
    answers = pd.DataFrame({"user_id": [1, 1], "movie_id": [10, 11], "rating": [5.0, 2.0]})
    assert add_labels(cands, answers, threshold=4.0)["label"].tolist() == [1, 0]


def test_leakage_guard():
    history = pd.DataFrame({"user_id": [1], "movie_id": [1], "rating": [5.0], "ts": [1000]})
    cands = pd.DataFrame({"user_id": [1], "movie_id": [1], "from_als": [True], "from_pop": [False]})
    movies = pd.DataFrame({"movie_id": [1], "title": ["X (2000)"]})
    with pytest.raises(AssertionError):
        add_features(cands, history, movies, GENRES, as_of_ts=1000, threshold=4.0)   # ts == as_of: not allowed