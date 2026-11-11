import tempfile
import time
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.data.movies import load_genres
from recsys.evaluation.evaluate import evaluate
from recsys.features.reranker import FEATURES
from recsys.models.popularity import PopularityRecommender
from recsys.models.reranker import LightGBMReranker
from recsys.tracking import git_info, setup_mlflow, summary_to_metrics


def main():
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])
    processed = cfg["paths"]["processed"]
    threshold = cfg["split"]["relevant_threshold"]
    k = 10

    learn = pd.read_parquet(processed / "features_2017.parquet")   # features up to 2016, answers 2017
    check = pd.read_parquet(processed / "features_2018.parquet")   # features up to 2017, answers 2018
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    genres = load_genres(movies)
    known_users = set(train["user_id"])
    n_catalog = train["movie_id"].nunique()
    val_users = val["user_id"].unique().tolist()
    popularity = PopularityRecommender(threshold=threshold, recent_days=90).fit(train)

    pd.set_option("display.float_format", "{:.4f}".format)
    for objective in ["binary", "lambdarank"]:
        with mlflow.start_run(run_name=f"reranker lightgbm {objective}"):
            start = time.time()
            model = LightGBMReranker(objective=objective, seed=cfg["seed"]).fit(learn)   # learns ONLY from 2017
            fit_seconds = time.time() - start

            # Hybrid: re-ranker for users with a shortlist, popularity for everyone else
            recs = model.recommend(check, k=k)
            fallback_users = [u for u in val_users if u not in recs]
            recs.update(popularity.recommend(fallback_users, k=k))

            per_user, summary = evaluate(recs, val, known_users, genres, n_catalog, k=k, threshold=threshold)
            importance = model.feature_importance()

            mlflow.log_params({
                "model": "reranker_lightgbm", "objective": objective,
                "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                "features": ",".join(FEATURES),
                "candidates": "als_top100 (2017 likes, alpha=10, reg=1.0) + popularity_90d_top100",
                "fallback": "popularity_90d", "k": k, "eval_set": "val",
            })
            mlflow.set_tags(git_info())
            mlflow.log_metrics(summary_to_metrics(summary))
            mlflow.log_metric("fit_seconds", fit_seconds)
            with tempfile.TemporaryDirectory() as tmp:
                per_user.to_parquet(Path(tmp) / "per_user_scores.parquet")
                importance.to_csv(Path(tmp) / "feature_importance.csv")
                mlflow.log_artifacts(tmp)

        print(f"\n=== Re-ranker ({objective}) | fit {fit_seconds:.0f}s ===")
        print(summary)
        print("\nFeature importance (share of total gain):")
        print((importance / importance.sum()).round(3).to_string())

    print("\nBars on returning users: popularity 0.1131 | ALS (shortlist setting) 0.1300 | best ALS 0.1315")


if __name__ == "__main__":
    main()