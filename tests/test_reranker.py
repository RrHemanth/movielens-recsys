import numpy as np
import pandas as pd
import pytest

from recsys.features.reranker import FEATURES
from recsys.models.reranker import LightGBMReranker


def make_table(seed, n_users=60, per_user=10):
    rng = np.random.default_rng(seed)
    n = n_users * per_user
    table = pd.DataFrame({f: rng.random(n) for f in FEATURES})   # every feature is random noise...
    table["user_id"] = np.repeat(np.arange(n_users), per_user)
    table["movie_id"] = np.tile(np.arange(per_user), n_users)
    table["label"] = (table["als_score"] > 0.7).astype(int)     # ...except: high taste match -> liked
    return table


@pytest.mark.parametrize("objective", ["binary", "lambdarank"])
def test_reranker_learns_that_taste_match_predicts_likes(objective):
    model = LightGBMReranker(objective=objective, n_estimators=50, min_child_samples=5).fit(make_table(seed=0))
    test = make_table(seed=1)   # fresh data the model hasn't seen
    top1 = model.recommend(test, k=1)

    labels = test.set_index(["user_id", "movie_id"])["label"]
    users_with_a_like = test.groupby("user_id")["label"].max().loc[lambda s: s == 1].index
    hit_rate = np.mean([labels[(u, top1[u][0])] for u in users_with_a_like])
    assert hit_rate > 0.9   # its first pick should almost always be a liked movie


def test_recommend_returns_k_per_user():
    model = LightGBMReranker(n_estimators=20, min_child_samples=5).fit(make_table(seed=0))
    recs = model.recommend(make_table(seed=1), k=3)
    assert all(len(movies) == 3 for movies in recs.values())