import pandas as pd
from pyspark.sql import functions as F

from recsys.data.split import to_unix_seconds
from recsys.models.als import ALSRecommender
from recsys.models.popularity import PopularityRecommender


def merge_candidate_lists(als_lists: dict, pop_lists: dict) -> pd.DataFrame:
    """Combine each user's ALS list and popularity list into one shortlist (no duplicates)."""
    als_rows = [(u, m, r) for u, movies in als_lists.items() for r, m in enumerate(movies, start=1)]
    pop_rows = [(u, m, r) for u, movies in pop_lists.items() for r, m in enumerate(movies, start=1)]
    als_df = pd.DataFrame(als_rows, columns=["user_id", "movie_id", "als_rank"])
    pop_df = pd.DataFrame(pop_rows, columns=["user_id", "movie_id", "pop_rank"])
    cands = als_df.merge(pop_df, on=["user_id", "movie_id"], how="outer")
    cands["from_als"] = cands["als_rank"].notna()
    cands["from_pop"] = cands["pop_rank"].notna()
    return cands.sort_values(["user_id", "movie_id"]).reset_index(drop=True)


def build_candidates(spark, ratings_paths: list, as_of: str, users: list, als_params: dict,
                     n_als: int = 100, n_pop: int = 100, threshold: float = 4.0, seed: int = 42) -> pd.DataFrame:
    """Build each user's shortlist using ONLY ratings before the pretend date `as_of`."""
    as_of_ts = to_unix_seconds(as_of)
    year_before_ts = to_unix_seconds(f"{int(as_of[:4]) - 1}{as_of[4:]}")   # e.g. 2017-01-01 -> 2016-01-01

    # History = everything before the pretend date: a Spark copy for ALS, a pandas copy for popularity.
    history_spark = spark.read.parquet(*[str(p) for p in ratings_paths]).filter(F.col("ts") < as_of_ts)
    history_pd = pd.concat([pd.read_parquet(p, filters=[("ts", "<", as_of_ts)]) for p in ratings_paths])

    als = ALSRecommender(**als_params, threshold=threshold, seed=seed,
                         train_start_ts=year_before_ts).fit(history_spark)
    als_lists = als.recommend(users, k=n_als)

    popularity = PopularityRecommender(threshold=threshold, recent_days=90).fit(history_pd)
    pop_lists = popularity.recommend(users, k=n_pop)

    return merge_candidate_lists(als_lists, pop_lists)


def candidate_recall(cands: pd.DataFrame, label_ratings: pd.DataFrame, threshold: float, users: list) -> pd.DataFrame:
    """Per user: of the movies they liked in the answer period, how many are on their shortlist?"""
    liked = label_ratings.loc[label_ratings["rating"] >= threshold, ["user_id", "movie_id"]]
    liked = liked[liked["user_id"].isin(set(users))]
    shortlist = cands[["user_id", "movie_id"]].assign(on_shortlist=True)
    found = liked.merge(shortlist, on=["user_id", "movie_id"], how="left")
    found["on_shortlist"] = found["on_shortlist"].fillna(False).astype(bool)
    per_user = found.groupby("user_id")["on_shortlist"].agg(n_liked="size", n_found="sum")
    per_user["recall"] = per_user["n_found"] / per_user["n_liked"]
    return per_user.reset_index()