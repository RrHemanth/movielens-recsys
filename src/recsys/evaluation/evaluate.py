import pandas as pd

from recsys.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    recall_at_k,
)


def relevant_movies(eval_ratings: pd.DataFrame, threshold: float) -> dict:
    """{user_id: set of movies they rated >= threshold in the evaluation period}."""
    liked = eval_ratings[eval_ratings["rating"] >= threshold]
    return liked.groupby("user_id")["movie_id"].apply(set).to_dict()


def evaluate(
    recommendations: dict,
    eval_ratings: pd.DataFrame,
    known_users: set,
    movie_genres: dict,
    n_catalog: int,
    k: int = 10,
    threshold: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a model's recommendations. Returns (per-user scores, summary by group)."""
    relevant = relevant_movies(eval_ratings, threshold)

    # 1. Score each user who has at least one relevant movie.
    rows = []
    for user_id, loved in relevant.items():
        if user_id not in recommendations:
            raise KeyError(f"No recommendations for user {user_id}. Every evaluated user needs a list.")
        recommended = recommendations[user_id]
        rows.append({
            "user_id": user_id,
            "group": "returning" if user_id in known_users else "new",
            "recall": recall_at_k(recommended, loved, k),
            "ndcg": ndcg_at_k(recommended, loved, k),
            "diversity": intra_list_diversity(recommended, movie_genres, k),
        })
    per_user = pd.DataFrame(rows)

    # 2. Summarize: all users, then each group separately.
    summary_rows = []
    groups = [("all", per_user)] + list(per_user.groupby("group"))
    for group_name, group_df in groups:
        lists = [recommendations[u] for u in group_df["user_id"]]
        summary_rows.append({
            "group": group_name,
            "n_users": len(group_df),
            "recall": group_df["recall"].mean(),
            "ndcg": group_df["ndcg"].mean(),
            "diversity": group_df["diversity"].mean(),
            "coverage": catalog_coverage(lists, n_catalog, k),
        })
    summary = pd.DataFrame(summary_rows).set_index("group")

    return per_user, summary