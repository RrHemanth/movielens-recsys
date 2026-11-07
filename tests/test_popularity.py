import pandas as pd

from recsys.models.popularity import PopularityRecommender

DAY = 24 * 3600


def make_train():
    # A: liked by 3 users long ago. B: liked by 2 users recently. C: liked by 1 user recently.
    # D: rated by 3 users, but nobody liked it (all 2 stars).
    rows = [
        (1, "A", 5.0, 0), (2, "A", 4.0, 0), (3, "A", 4.5, 0),
        (4, "B", 5.0, 1000 * DAY), (5, "B", 4.0, 1000 * DAY),
        (1, "C", 4.0, 1000 * DAY),
        (2, "D", 2.0, 1000 * DAY), (3, "D", 2.0, 1000 * DAY), (4, "D", 2.0, 1000 * DAY),
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "ts"])


def test_all_time_ranking_counts_only_liked_ratings():
    model = PopularityRecommender().fit(make_train())
    # D was rated 3 times but never liked, so it isn't ranked at all.
    assert model.ranking == ["A", "B", "C"]


def test_recent_ranking_ignores_old_ratings():
    model = PopularityRecommender(recent_days=30).fit(make_train())
    # A's likes are all ~1000 days old, so only B and C count.
    assert model.ranking == ["B", "C"]


def test_skips_movies_the_user_already_rated():
    model = PopularityRecommender().fit(make_train())
    recs = model.recommend([1, 4], k=2)
    assert recs[1] == ["B"]        # user 1 already rated A and C
    assert recs[4] == ["A", "C"]   # user 4 already rated B (and D)


def test_new_user_gets_plain_top_k():
    model = PopularityRecommender().fit(make_train())
    recs = model.recommend([99], k=2)  # user 99 isn't in training
    assert recs[99] == ["A", "B"]


def test_ties_are_broken_the_same_way_every_time():
    train = pd.DataFrame(
        [(1, "Y", 5.0, 0), (2, "X", 5.0, 0)],
        columns=["user_id", "movie_id", "rating", "ts"],
    )
    model = PopularityRecommender().fit(train)
    assert model.ranking == ["X", "Y"]  # same count -> ordered by movie_id