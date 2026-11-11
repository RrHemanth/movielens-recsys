import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.evaluation.stats import bootstrap_ci, compare_paired
from recsys.tracking import setup_mlflow

RUNS = {
    "popularity": "popularity last 90 days",
    "als_best": "als from 2017 rank=64 reg=0.1 alpha=1.0 iters=10",
    "als_shortlist": "als from 2017 rank=64 reg=1.0 alpha=10.0 iters=10",
    "rerank_binary": "reranker lightgbm binary",
    "rerank_lambdarank": "reranker lightgbm lambdarank",
}
COMPARISONS = [   # (A, B): is B better than A?
    ("popularity", "als_best"),
    ("als_shortlist", "als_best"),
    ("als_shortlist", "rerank_binary"),
    ("als_best", "rerank_binary"),
    ("rerank_lambdarank", "rerank_binary"),
]


def returning_ndcg(run_name: str, experiment: str) -> pd.Series:
    """Per-user NDCG for returning users, from the run's saved per_user_scores.parquet."""
    runs = mlflow.search_runs(experiment_names=[experiment],
                              filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"])
    if runs.empty:
        raise ValueError(f"No MLflow run named {run_name!r}")
    path = mlflow.artifacts.download_artifacts(run_id=runs.iloc[0]["run_id"],
                                               artifact_path="per_user_scores.parquet")
    df = pd.read_parquet(path)
    return df[df["group"] == "returning"].set_index("user_id")["ndcg"].sort_index()


def main():
    cfg = load_config()
    experiment = cfg["mlflow"]["experiment"]
    setup_mlflow(experiment)
    scores = {name: returning_ndcg(run, experiment) for name, run in RUNS.items()}

    # Paired comparisons need the SAME users in every model.
    users = scores["popularity"].index
    assert all(s.index.equals(users) for s in scores.values()), "Models were scored on different users"
    print(f"Returning users: {len(users):,}\n")

    print("NDCG@10 with 95% bootstrap confidence intervals:")
    for name, s in scores.items():
        mean, low, high = bootstrap_ci(s.to_numpy())
        print(f"  {name:18s} {mean:.4f}  [{low:.4f}, {high:.4f}]")

    print("\nPaired comparisons (B - A, same users):")
    print(f"  {'A':18s} {'B':18s} {'diff':>8s} {'95% CI':>20s} {'p-value':>8s} {'B better':>9s} {'B worse':>8s}")
    for a, b in COMPARISONS:
        r = compare_paired(scores[a].to_numpy(), scores[b].to_numpy())
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"  {a:18s} {b:18s} {r['mean_diff']:+8.4f} {ci:>20s} {r['p_value']:8.4f} "
              f"{r['share_better']:9.1%} {r['share_worse']:8.1%}")


if __name__ == "__main__":
    main()