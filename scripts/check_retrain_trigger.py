import os

import pandas as pd

from recsys.config import load_config
from recsys.monitoring.drift import should_retrain


def main():
    cfg = load_config()
    report = pd.read_csv(cfg["paths"]["processed"] / "drift_report.csv")
    latest = report.iloc[-1].to_dict()
    retrain, worst = should_retrain(latest)
    print(f"Latest month {latest['month']}: worst signal {worst} = {latest[worst]:.3f} -> retrain: {retrain}")

    # Inside GitHub Actions, pass the decision on to later steps
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"retrain={'true' if retrain else 'false'}\n")


if __name__ == "__main__":
    main()