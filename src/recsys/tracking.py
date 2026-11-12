import subprocess

import mlflow
import pandas as pd

from recsys.config import REPO_ROOT


def setup_mlflow(experiment_name: str) -> None:
    """Store MLflow records in the repo (mlflow.db + mlruns/) and select the experiment."""
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment_name)


def git_info() -> dict:
    """Which commit is the code at, and are there uncommitted changes?"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        return {"git_commit": commit, "git_uncommitted_changes": str(bool(changes))}
    except (subprocess.CalledProcessError, OSError):
        return {"git_commit": "unknown", "git_uncommitted_changes": "unknown"}


def summary_to_metrics(summary: pd.DataFrame) -> dict:
    """Turn the evaluate() summary table into flat names, e.g. {'ndcg_returning': 0.113, ...}."""
    return {
        f"{metric}_{group}": float(value)
        for group, row in summary.iterrows()
        for metric, value in row.items()
    }