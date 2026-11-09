import pandas as pd

from recsys.tracking import summary_to_metrics


def test_summary_to_metrics_names_and_values():
    summary = pd.DataFrame(
        {"n_users": [10, 4], "ndcg": [0.25, 0.5]},
        index=pd.Index(["all", "new"], name="group"),
    )
    assert summary_to_metrics(summary) == {
        "n_users_all": 10.0,
        "ndcg_all": 0.25,
        "n_users_new": 4.0,
        "ndcg_new": 0.5,
    }