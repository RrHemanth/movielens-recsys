import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.data.split import to_unix_seconds
from recsys.features.reranker import movie_year
from recsys.models.similarity import load_vectors
from recsys.monitoring.drift import drift_report, psi_status, window_signals
from recsys.tracking import git_info, setup_mlflow

PSI_COLUMNS = ["psi_activity", "psi_movie_age", "psi_rating", "psi_unknown_movie"]


def main():
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])
    processed = cfg["paths"]["processed"]

    train = pd.read_parquet(processed / "train.parquet")
    later = pd.concat([pd.read_parquet(processed / "val.parquet"),
                       pd.read_parquet(processed / "test.parquet")], ignore_index=True)
    movies = pd.read_parquet(cfg["paths"]["interim"] / "movies.parquet")
    years = {m: movie_year(t) for m, t in zip(movies["movie_id"], movies["title"])}

    # The model being monitored: built as of 2018-01-01 (ALS on 2017 likes), never retrained here.
    known_movies = set(load_vectors(processed / "als" / "movie_vectors.parquet")[0].tolist())
    known_users = set(train["user_id"])

    reference_ratings = train[train["ts"] >= to_unix_seconds("2017-01-01")]   # the model's training year
    reference = window_signals(reference_ratings, years, known_movies)
    print(f"Reference: {len(reference_ratings):,} ratings from 2017 | "
          f"{reference['unknown_movie'].mean():.1%} on movies without a vector\n")

    rows = []
    later["month"] = pd.to_datetime(later["ts"], unit="s").dt.to_period("M")
    for month, ratings in later.groupby("month"):
        report = drift_report(reference, window_signals(ratings, years, known_movies))
        report["new_user_share"] = float((~ratings["user_id"].isin(known_users)).mean())
        report["n_ratings"] = len(ratings)
        report["status"] = psi_status(max(report[c] for c in PSI_COLUMNS))
        rows.append({"month": str(month), **report})
    table = pd.DataFrame(rows)

    out = processed / "drift_report.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    print(table[["month", "n_ratings", *PSI_COLUMNS, "unknown_movie_share", "new_user_share", "status"]]
          .to_string(index=False))
    print(f"\nSaved -> {out.name}")

    with mlflow.start_run(run_name="drift replay (model as of 2018-01-01)"):
        mlflow.log_params({"reference": "2017 ratings", "model_as_of": "2018-01-01", "psi_thresholds": "0.1 / 0.25"})
        mlflow.set_tags(git_info())
        for step, row in table.iterrows():   # one point per month -> line charts in the MLflow UI
            mlflow.log_metrics({c: row[c] for c in [*PSI_COLUMNS, "unknown_movie_share", "new_user_share"]},
                               step=step)
        with tempfile.TemporaryDirectory() as tmp:
            table.to_csv(Path(tmp) / "drift_report.csv", index=False)
            mlflow.log_artifacts(tmp)


if __name__ == "__main__":
    main()