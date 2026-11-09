import pandas as pd
import pytest

from recsys.models.als import ALSRecommender
from recsys.spark import get_spark


@pytest.fixture(scope="module")
def spark():
    session = get_spark("test-als", driver_memory="1g")
    yield session
    session.stop()


def make_train(spark):
    rows = []
    # Group A (users 1-4) likes movies 10, 11, 12 -- except user 1 hasn't seen 12 yet.
    for user in [1, 2, 3, 4]:
        rows += [(user, 10, 5.0), (user, 11, 4.5)]
    for user in [2, 3, 4]:
        rows.append((user, 12, 5.0))
    # Group B (users 5-8) likes movies 20, 21, 22.
    for user in [5, 6, 7, 8]:
        rows += [(user, 20, 5.0), (user, 21, 4.5), (user, 22, 4.0)]
    return spark.createDataFrame(pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"]))


def test_recommends_what_similar_users_liked(spark):
    model = ALSRecommender(rank=2, reg_param=0.01, alpha=10.0, max_iter=15).fit(make_train(spark))
    # User 1 is in group A, and movie 12 is the group-A movie they haven't seen.
    assert model.recommend([1], k=1)[1] == [12]


def test_never_recommends_already_rated_movies(spark):
    model = ALSRecommender(rank=2, reg_param=0.01, alpha=10.0, max_iter=15).fit(make_train(spark))
    recs = model.recommend([1], k=4)[1]
    assert 10 not in recs and 11 not in recs


def test_unknown_users_are_left_out(spark):
    model = ALSRecommender(rank=2, reg_param=0.01, alpha=10.0, max_iter=15).fit(make_train(spark))
    assert model.recommend([1, 99], k=2).keys() == {1}  # user 99 never appeared in training