import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class ALSRecommender:
    """Implicit-feedback ALS: learns taste vectors from 'liked' ratings (rating >= threshold)."""

    def __init__(self, rank=64, reg_param=0.1, alpha=10.0, max_iter=10, threshold=4.0, seed=42, train_start_ts=None):
        self.rank = rank              # how many taste numbers per user/movie
        self.reg_param = reg_param    # how strongly to keep taste numbers small (regularization)
        self.alpha = alpha            # how much more a "liked" counts than an empty cell
        self.max_iter = max_iter      # rounds of back-and-forth (users, then movies)
        self.threshold = threshold    # rating >= this counts as "liked"
        self.seed = seed              # fixes randomness so reruns give the same result
        self.train_start_ts = train_start_ts  # learn only from ratings at/after this time (None = all)

    def fit(self, train: DataFrame) -> "ALSRecommender":
        """Learn taste vectors from a Spark DataFrame with user_id, movie_id, rating."""
                # Learn taste only from the chosen time window...
        learn_from = train
        if self.train_start_ts is not None:
            learn_from = train.filter(F.col("ts") >= self.train_start_ts)
        liked = (
            learn_from.filter(F.col("rating") >= self.threshold)
            .select("user_id", "movie_id", F.lit(1.0).alias("liked"))
        )
        als = ALS(
            userCol="user_id", itemCol="movie_id", ratingCol="liked",
            implicitPrefs=True,                       # the "liked or not" version
            rank=self.rank, regParam=self.reg_param, alpha=self.alpha,
            maxIter=self.max_iter, seed=self.seed,
        )
        model = als.fit(liked)

        # Copy the learned vectors out of Spark into NumPy arrays.
        users = model.userFactors.toPandas()      # columns: id, features
        movies = model.itemFactors.toPandas()
        self.user_index = {int(u): i for i, u in enumerate(users["id"])}
        self.user_vectors = np.array(users["features"].tolist(), dtype=np.float32)
        self.movie_ids = movies["id"].to_numpy()
        self.movie_index = {int(m): i for i, m in enumerate(self.movie_ids)}
        self.movie_vectors = np.array(movies["features"].tolist(), dtype=np.float32)

        self.train_pairs = train.select("user_id", "movie_id")  # to know what each user has seen
        return self

    def recommend(self, user_ids: list, k: int = 10, batch_size: int = 1000) -> dict:
        """Top-k unseen movies for each user ALS knows. Users it never saw are left out."""
        known = [u for u in user_ids if u in self.user_index]
        seen = self._seen_movies(known)

        recommendations = {}
        for start in range(0, len(known), batch_size):
            batch = known[start:start + batch_size]
            vectors = self.user_vectors[[self.user_index[u] for u in batch]]
            scores = vectors @ self.movie_vectors.T   # every user in batch x every movie

            for row, user_id in enumerate(batch):
                user_scores = scores[row]
                for movie in seen.get(user_id, ()):   # never recommend what they've rated
                    pos = self.movie_index.get(int(movie))
                    if pos is not None:
                        user_scores[pos] = -np.inf
                top = np.argsort(-user_scores)[:k]    # highest scores first
                top = top[np.isfinite(user_scores[top])]
                recommendations[user_id] = self.movie_ids[top].tolist()
        return recommendations

    def _seen_movies(self, user_ids: list) -> dict:
        """{user_id: set of movies they rated in training}, only for the given users."""
        if not user_ids:
            return {}
        history = self.train_pairs.filter(F.col("user_id").isin(user_ids)).toPandas()
        return history.groupby("user_id")["movie_id"].apply(set).to_dict()