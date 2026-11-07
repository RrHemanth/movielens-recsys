import pandas as pd


class PopularityRecommender:
    """Recommend the most-liked movies from training, skipping ones the user has already rated."""

    def __init__(self, threshold: float = 4.0, recent_days: int | None = None):
        self.threshold = threshold        # a rating >= this counts as "liked"
        self.recent_days = recent_days    # None = all of training; e.g. 365 = last year only
        self.ranking = []                 # movie_ids, most popular first
        self.train_pairs = None           # (user_id, movie_id) pairs, to know what each user has seen

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        data = train
        if self.recent_days is not None:
            # Keep only the last `recent_days` days before the end of training.
            cutoff = train["ts"].max() - self.recent_days * 24 * 3600
            data = train[train["ts"] >= cutoff]

        liked = data[data["rating"] >= self.threshold]
        counts = (
            liked.groupby("movie_id").size().reset_index(name="n_likes")
            # Most likes first; ties broken by movie_id so results are always the same.
            .sort_values(["n_likes", "movie_id"], ascending=[False, True])
        )
        self.ranking = counts["movie_id"].tolist()
        self.train_pairs = train[["user_id", "movie_id"]]
        return self

    def recommend(self, user_ids: list, k: int = 10) -> dict:
        """Return {user_id: [top k movie_ids they haven't rated in training]}."""
        # Only look up history for the users we're asked about (much faster than all 150k users).
        history = self.train_pairs[self.train_pairs["user_id"].isin(set(user_ids))]
        seen = history.groupby("user_id")["movie_id"].apply(set).to_dict()

        recommendations = {}
        for user_id in user_ids:
            already_seen = seen.get(user_id, set())   # new users have seen nothing
            top = []
            for movie in self.ranking:
                if movie not in already_seen:
                    top.append(movie)
                    if len(top) == k:
                        break
            recommendations[user_id] = top
        return recommendations