import numpy as np
import pandas as pd

from recsys.models.cold_start import FoldInRecommender, blend_ranks, onboarding_split, top_unseen


def test_onboarding_split_uses_time_order_and_drops_short_users():
    ratings = pd.DataFrame({
        "user_id": [1] * 7 + [2] * 5,
        "movie_id": list(range(10, 17)) + list(range(20, 25)),
        "rating": [4.0] * 12,
        "ts": [7, 6, 5, 4, 3, 2, 1] + [1, 2, 3, 4, 5],   # user 1's ratings arrive in reverse movie order
    })
    onboard, rest = onboarding_split(ratings, n_onboard=5)
    assert sorted(onboard["movie_id"]) == [12, 13, 14, 15, 16]   # user 1's five EARLIEST ratings
    assert sorted(rest["movie_id"]) == [10, 11]
    assert 2 not in set(onboard["user_id"]) | set(rest["user_id"])   # only 5 ratings -> nothing to grade


def test_fold_in_recommends_like_nina():
    # (fun, intense): Toy Story, Up, Finding Nemo, Monsters Inc., Heat
    movie_ids = [1, 2, 3, 4, 5]
    vectors = np.array([[0.9, 0.1], [0.8, 0.2], [0.85, 0.15], [0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    onboard = pd.DataFrame({"user_id": [7, 7, 7], "movie_id": [1, 2, 3], "rating": [5.0, 4.5, 5.0]})
    recs = FoldInRecommender(movie_ids, vectors).recommend(onboard, k=1)
    assert recs[7] == [4]   # Monsters Inc., not Heat, and never her onboarding movies


def test_fold_in_skips_users_with_no_liked_movies():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    onboard = pd.DataFrame({"user_id": [8, 8], "movie_id": [1, 2], "rating": [2.0, 3.0]})
    assert FoldInRecommender([1, 2], vectors).recommend(onboard, k=1) == {}


def test_blend_ranks_example():
    # Taste list: M, I, x1, x2   |   Popular list: N, y1, I, y2   |   missing rank = 5
    # I: (2+3)/2 = 2.5   M: (1+5)/2 = 3   N: (5+1)/2 = 3   y1: 3.5   x1: 4
    blended = blend_ranks(["M", "I", "x1", "x2"], ["N", "y1", "I", "y2"], k=3, depth=4)
    assert blended == ["I", "M", "N"]   # M and N tie at 3; M wins on taste rank


def test_top_unseen():
    assert top_unseen([1, 2, 3, 4], seen={1, 3}, k=2) == [2, 4]