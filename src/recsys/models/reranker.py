import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.features.reranker import FEATURES


class LightGBMReranker:
    """Re-ranks each user's shortlist with LightGBM, as 'binary' (classification) or 'lambdarank' (learning to rank)."""

    def __init__(self, objective="lambdarank", n_estimators=300, learning_rate=0.05,
                 num_leaves=31, min_child_samples=20, seed=42):
        self.objective = objective
        settings = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                        min_child_samples=min_child_samples, random_state=seed, verbose=-1)
        if objective == "lambdarank":
            self.model = lgb.LGBMRanker(objective="lambdarank", **settings)
        else:
            self.model = lgb.LGBMClassifier(objective="binary", **settings)

    def fit(self, table: pd.DataFrame) -> "LightGBMReranker":
        """Learn from rows of (user, candidate, features, label)."""
        table = table.sort_values("user_id", kind="stable")   # each user's rows must sit together
        if self.objective == "lambdarank":
            group_sizes = table.groupby("user_id", sort=True).size().to_numpy()   # rows per user, in order
            self.model.fit(table[FEATURES], table["label"], group=group_sizes)
        else:
            self.model.fit(table[FEATURES], table["label"])
        return self

    def score(self, table: pd.DataFrame) -> np.ndarray:
        if self.objective == "lambdarank":
            return self.model.predict(table[FEATURES])
        return self.model.predict_proba(table[FEATURES])[:, 1]   # probability of "liked"

    def recommend(self, table: pd.DataFrame, k: int = 10) -> dict:
        """Score every candidate, then keep each user's k highest."""
        scored = table[["user_id", "movie_id"]].assign(score=self.score(table))
        top = scored.sort_values(["user_id", "score"], ascending=[True, False], kind="stable").groupby("user_id").head(k)
        return top.groupby("user_id")["movie_id"].apply(list).to_dict()

    def feature_importance(self) -> pd.Series:
        """How much each feature improved the model, summed over all trees ('gain')."""
        gain = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(gain, index=FEATURES).sort_values(ascending=False)