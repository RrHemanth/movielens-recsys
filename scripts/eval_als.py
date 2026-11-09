import argparse
import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.data.movies import load_genres
from recsys.evaluation.evaluate import evaluate
from recsys.models.als import ALSRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.spark import get_spark
from recsys.tracking import git_info, setup_mlflow, summary_to_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train ALS on train, evaluate on 2018 validation.")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--iters", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])
    threshold = cfg["split"]["relevant_threshold"]
    processed = cfg["paths"]["processed"]
    k = 10

    # pandas copies: for evaluation and for the popularity fallback
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    genres = load_genres(movies)
    known_users = set(train["user_id"])
    n_catalog = train["movie_id"].nunique()
    val_users = val["user_id"].unique().tolist()

    run_name = f"als rank={args.rank} reg={args.reg} alpha={args.alpha} iters={args.iters}"
    with mlflow.start_run(run_name=run_name):
        # 1. Train ALS with Spark
        spark = get_spark("als")
        start = time.time()
        als = ALSRecommender(
            rank=args.rank, reg_param=args.reg, alpha=args.alpha, max_iter=args.iters,
            threshold=threshold, seed=cfg["seed"],
        ).fit(spark.read.parquet(str(processed / "train.parquet")))
        fit_seconds = time.time() - start

        # 2. Hybrid: ALS for users it knows, 90-day popularity for everyone else
        start = time.time()
        recs = als.recommend(val_users, k=k)
        fallback_users = [u for u in val_users if u not in recs]
        popularity = PopularityRecommender(threshold=threshold, recent_days=90).fit(train)
        recs.update(popularity.recommend(fallback_users, k=k))
        recommend_seconds = time.time() - start
        spark.stop()

        n_new = sum(1 for u in fallback_users if u not in known_users)
        n_returning_fallback = len(fallback_users) - n_new

        # 3. Score
        per_user, summary = evaluate(recs, val, known_users, genres, n_catalog, k=k, threshold=threshold)

        # 4. Record everything in MLflow
        mlflow.log_params({
            "model": "als_hybrid",
            "rank": args.rank, "reg_param": args.reg, "alpha": args.alpha, "max_iter": args.iters,
            "fallback": "popularity_90d",
            "k": k, "relevant_threshold": threshold, "eval_set": "val",
            "val_start": cfg["split"]["val_start"], "test_start": cfg["split"]["test_start"],
        })
        mlflow.set_tags(git_info())
        mlflow.log_metrics(summary_to_metrics(summary))
        mlflow.log_metrics({
            "fit_seconds": fit_seconds,
            "recommend_seconds": recommend_seconds,
            "n_fallback_new_users": n_new,
            "n_fallback_returning_users": n_returning_fallback,
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_user_scores.parquet"
            per_user.to_parquet(path)
            mlflow.log_artifact(str(path))

    pd.set_option("display.float_format", "{:.4f}".format)
    print(f"\n=== {run_name} ===")
    print(f"Fit: {fit_seconds:.0f}s | recommend: {recommend_seconds:.0f}s")
    print(f"Fallback to popularity: {n_new:,} new users + {n_returning_fallback:,} returning users ALS couldn't score")
    print(summary)
    print("\nBar to beat (popularity, 90 days): ndcg_returning = 0.1131")


if __name__ == "__main__":
    main()