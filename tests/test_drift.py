import numpy as np
import pytest

from recsys.monitoring.drift import (
    ks_statistic,
    psi_categorical,
    psi_from_shares,
    psi_numeric,
    psi_status,
    should_retrain,
)


def test_psi_worked_example():
    # The low/medium/high rating example: 20/30/50% -> 25/30/45%
    assert psi_from_shares([0.20, 0.30, 0.50], [0.25, 0.30, 0.45]) == pytest.approx(0.0164, abs=0.0005)


def test_identical_distributions_have_zero_psi():
    values = np.arange(1000)
    assert psi_numeric(values, values) == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_new_sample_is_stable_and_shifted_one_drifts():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 20_000)
    assert psi_numeric(reference, rng.normal(0, 1, 20_000)) < 0.1    # just a new sample: stable
    assert psi_numeric(reference, rng.normal(1, 1, 20_000)) > 0.25   # mean moved by 1 std: drift


def test_categorical_handles_a_brand_new_category():
    assert psi_categorical(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["c"] * 50) > 0.25


def test_ks_statistic_extremes():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_statistic([1, 2, 3], [10, 11, 12]) == 1.0


def test_status_thresholds():
    assert [psi_status(v) for v in (0.05, 0.15, 0.4)] == ["stable", "watch", "drift"]

def test_retrain_when_any_signal_reaches_watch():
    row = {"psi_activity": 0.02, "psi_movie_age": 0.01, "psi_rating": 0.0, "psi_unknown_movie": 0.12}
    assert should_retrain(row) == (True, "psi_unknown_movie")


def test_no_retrain_when_everything_is_stable():
    row = {"psi_activity": 0.02, "psi_movie_age": 0.01, "psi_rating": 0.0, "psi_unknown_movie": 0.05}
    assert should_retrain(row)[0] is False