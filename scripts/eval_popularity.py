import tempfile
import time
from pathlib import Path
from recsys.data.movies import load_genres

import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.evaluation.evaluate import evaluate
from recsys.models.popularity import PopularityRecommender
from recsys.tracking import git_info, setup_mlflow, summary_to_metrics


def load_genres(movies: pd.DataFrame) -> dict:
    """{movie_id: set of genres}. '(no genres listed)' becomes an empty set."""
    genres = {}
    for movie_id, genre_text in zip(movies["movie_id"], movies["genres"]):
        if genre_text == "(no genres listed)":
            genres[movie_id] = set()
        else:
            genres[movie_id] = set(genre_text.split("|"))
    return genres


def main():
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])  # MLflow: choose where to record runs

    train = pd.read_parquet(cfg["paths"]["processed"] / "train.parquet")
    val = pd.read_parquet(cfg["paths"]["processed"] / "val.parquet")
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")

    genres = load_genres(movies)
    titles = dict(zip(movies["movie_id"], movies["title"]))
    known_users = set(train["user_id"])
    n_catalog = train["movie_id"].nunique()
    val_users = val["user_id"].unique().tolist()
    threshold = cfg["split"]["relevant_threshold"]
    k = 10

    print(f"Train ratings: {len(train):,} | catalog: {n_catalog:,} movies | 2018 users: {len(val_users):,}")
    pd.set_option("display.float_format", "{:.4f}".format)

    for recent_days in [None, 365, 90]:
        name = "all-time" if recent_days is None else f"last {recent_days} days"

        with mlflow.start_run(run_name=f"popularity {name}"):  # MLflow: start recording
            start = time.time()
            model = PopularityRecommender(threshold=threshold, recent_days=recent_days).fit(train)
            recs = model.recommend(val_users, k=k)
            per_user, summary = evaluate(recs, val, known_users, genres, n_catalog, k=k, threshold=threshold)

            # MLflow: the settings that define this run
            mlflow.log_params({
                "model": "popularity",
                "recent_days": recent_days if recent_days is not None else "all",
                "k": k,
                "relevant_threshold": threshold,
                "eval_set": "val",
                "val_start": cfg["split"]["val_start"],
                "test_start": cfg["split"]["test_start"],
            })
            # MLflow: which code produced it
            mlflow.set_tags(git_info())
            # MLflow: all the scores, e.g. ndcg_returning, recall_new, coverage_all
            mlflow.log_metrics(summary_to_metrics(summary))
            mlflow.log_metric("runtime_seconds", time.time() - start)
            # MLflow: save the per-user scores as a file (needed for confidence intervals later)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "per_user_scores.parquet"
                per_user.to_parquet(path)
                mlflow.log_artifact(str(path))

        print(f"\n=== Popularity ({name}) | {time.time() - start:.1f}s ===")
        print("Top 5 movies:", [titles[m] for m in model.ranking[:5]])
        print(summary)


if __name__ == "__main__":
    main()