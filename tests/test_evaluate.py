import pandas as pd
import pytest

from recsys.evaluation.evaluate import evaluate


def make_example():
    eval_ratings = pd.DataFrame({
        "user_id":  [1,   1,   1,   2,   3],
        "movie_id": ["A", "C", "B", "D", "E"],
        "rating":   [5.0, 4.0, 2.0, 4.5, 2.0],
    })
    recommendations = {1: ["A", "B"], 2: ["X", "D"], 3: ["A", "B"]}
    known_users = {1, 3}  # user 2 is new
    genres = {"A": {"Comedy"}, "B": {"Drama"}, "X": {"Comedy"}, "D": {"Comedy"}}
    return eval_ratings, recommendations, known_users, genres


def test_per_user_scores_and_skipping():
    eval_ratings, recs, known, genres = make_example()
    per_user, _ = evaluate(recs, eval_ratings, known, genres, n_catalog=10, k=2)

    # User 3 has no 4+ ratings, so only users 1 and 2 are scored.
    assert sorted(per_user["user_id"]) == [1, 2]

    row1 = per_user.set_index("user_id").loc[1]
    assert row1["group"] == "returning"
    assert row1["recall"] == pytest.approx(0.5)
    assert row1["ndcg"] == pytest.approx(0.61, abs=0.01)
    assert row1["diversity"] == pytest.approx(1.0)   # Comedy vs Drama

    row2 = per_user.set_index("user_id").loc[2]
    assert row2["group"] == "new"
    assert row2["recall"] == pytest.approx(1.0)
    assert row2["ndcg"] == pytest.approx(0.63, abs=0.01)
    assert row2["diversity"] == pytest.approx(0.0)   # Comedy vs Comedy


def test_summary_by_group():
    eval_ratings, recs, known, genres = make_example()
    _, summary = evaluate(recs, eval_ratings, known, genres, n_catalog=10, k=2)

    assert summary.loc["all", "n_users"] == 2
    assert summary.loc["all", "recall"] == pytest.approx(0.75)      # (0.5 + 1.0) / 2
    assert summary.loc["returning", "recall"] == pytest.approx(0.5)
    assert summary.loc["new", "recall"] == pytest.approx(1.0)
    assert summary.loc["all", "coverage"] == pytest.approx(0.4)     # A, B, X, D -> 4 of 10
    assert summary.loc["returning", "coverage"] == pytest.approx(0.2)  # A, B -> 2 of 10


def test_missing_recommendations_raises():
    eval_ratings, recs, known, genres = make_example()
    del recs[2]  # pretend the model couldn't produce a list for the new user
    with pytest.raises(KeyError):
        evaluate(recs, eval_ratings, known, genres, n_catalog=10, k=2)