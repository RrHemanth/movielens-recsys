import mlflow
import pandas as pd

from recsys.config import load_config
from recsys.tracking import setup_mlflow

COLUMNS = {
    "tags.mlflow.runName": "run",
    "metrics.ndcg_returning": "ndcg_ret",
    "metrics.recall_returning": "recall_ret",
    "metrics.coverage_returning": "coverage_ret",
    "metrics.n_fallback_returning_users": "fallback_ret",
}


def main():
    cfg = load_config()
    setup_mlflow(cfg["mlflow"]["experiment"])
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    table = runs[list(COLUMNS)].rename(columns=COLUMNS).sort_values("ndcg_ret", ascending=False)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()