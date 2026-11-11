import pandas as pd

from recsys.candidates import candidate_recall, merge_candidate_lists


def test_merge_keeps_each_movie_once_and_remembers_sources():
    cands = merge_candidate_lists({1: [10, 11]}, {1: [11, 12]})
    assert cands["movie_id"].tolist() == [10, 11, 12]
    by_movie = cands.set_index("movie_id")
    assert by_movie.loc[11, "from_als"] and by_movie.loc[11, "from_pop"]   # on both lists
    assert by_movie.loc[11, "als_rank"] == 2 and by_movie.loc[11, "pop_rank"] == 1
    assert not by_movie.loc[10, "from_pop"]


def test_user_without_als_vector_still_gets_popular_candidates():
    cands = merge_candidate_lists({}, {2: [12, 13]})
    assert cands["movie_id"].tolist() == [12, 13]
    assert not cands["from_als"].any()


def test_candidate_recall_counts_only_liked_movies():
    cands = pd.DataFrame({"user_id": [1, 1, 1], "movie_id": [10, 11, 12]})
    answers = pd.DataFrame({"user_id": [1, 1, 1], "movie_id": [11, 99, 12], "rating": [5.0, 4.0, 2.0]})
    # Liked (4+): 11 and 99. Movie 12 was rated 2, so it doesn't count. Found: 11 -> 1 of 2.
    per_user = candidate_recall(cands, answers, threshold=4.0, users=[1])
    assert per_user.loc[0, "recall"] == 0.5


def test_user_with_no_candidates_has_zero_recall():
    cands = pd.DataFrame({"user_id": [1], "movie_id": [10]})
    answers = pd.DataFrame({"user_id": [2], "movie_id": [10], "rating": [5.0]})
    per_user = candidate_recall(cands, answers, threshold=4.0, users=[1, 2])
    assert per_user.set_index("user_id").loc[2, "recall"] == 0.0