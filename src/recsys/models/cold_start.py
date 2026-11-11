import numpy as np
import pandas as pd


def onboarding_split(ratings: pd.DataFrame, n_onboard: int = 5):
    """Per user, in time order: the first n_onboard ratings are 'onboarding', the rest are the answer key.
    Users with n_onboard or fewer ratings are dropped (nothing left to grade them on)."""
    ordered = ratings.sort_values(["user_id", "ts", "movie_id"], kind="stable")
    position = ordered.groupby("user_id").cumcount()                      # 0, 1, 2, ... per user
    n_ratings = ordered.groupby("user_id")["movie_id"].transform("size")
    keep = n_ratings > n_onboard
    return ordered[keep & (position < n_onboard)], ordered[keep & (position >= n_onboard)]


def top_unseen(ranking: list, seen: set, k: int) -> list:
    """First k movies from a ranking that the user hasn't already rated."""
    result = []
    for movie in ranking:
        if movie not in seen:
            result.append(movie)
            if len(result) == k:
                break
    return result


def blend_ranks(taste_list: list, popular_list: list, k: int, depth: int) -> list:
    """Average each movie's rank in the two lists (missing = depth + 1); lowest average first.
    Ties go to the movie with the better taste rank."""
    taste_rank = {m: i + 1 for i, m in enumerate(taste_list[:depth])}
    popular_rank = {m: i + 1 for i, m in enumerate(popular_list[:depth])}
    missing = depth + 1
    movies = set(taste_rank) | set(popular_rank)
    average = {m: (taste_rank.get(m, missing) + popular_rank.get(m, missing)) / 2 for m in movies}
    ordered = sorted(movies, key=lambda m: (average[m], taste_rank.get(m, missing), m))
    return ordered[:k]


class FoldInRecommender:
    """New users: build a taste vector by averaging the vectors of the movies they liked during onboarding."""

    def __init__(self, movie_ids, movie_vectors: np.ndarray, threshold: float = 4.0):
        self.movie_ids = np.asarray(movie_ids)
        self.movie_vectors = movie_vectors
        self.movie_index = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.threshold = threshold

    def recommend(self, onboard: pd.DataFrame, k: int = 10) -> dict:
        """{user_id: top-k movies}. Users with no liked onboarding movie that has a vector are left out."""
        recommendations = {}
        for user_id, rows in onboard.groupby("user_id"):
            liked = rows.loc[rows["rating"] >= self.threshold, "movie_id"]
            positions = [self.movie_index[int(m)] for m in liked if int(m) in self.movie_index]
            if not positions:
                continue                                          # nothing to build a taste from
            taste = self.movie_vectors[positions].mean(axis=0)    # the "average of what you liked"
            scores = self.movie_vectors @ taste                   # dot product with every movie
            for m in rows["movie_id"]:                            # never recommend onboarding movies
                pos = self.movie_index.get(int(m))
                if pos is not None:
                    scores[pos] = -np.inf
            top = np.argsort(-scores)[:k]
            recommendations[user_id] = self.movie_ids[top].tolist()
        return recommendations