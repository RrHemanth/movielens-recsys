import numpy as np
import pytest

from recsys.evaluation.stats import bootstrap_ci, compare_paired


def test_constant_values_have_zero_width_interval():
    mean, low, high = bootstrap_ci(np.full(100, 0.3))
    assert mean == low == high == pytest.approx(0.3)


def test_interval_matches_textbook_formula():
    # 2,000 values of 0 or 1 -> mean 0.5; textbook 95% interval is about 0.5 +/- 1.96 * 0.5 / sqrt(2000) = +/- 0.022
    values = np.array([0.0, 1.0] * 1000)
    mean, low, high = bootstrap_ci(values)
    assert mean == pytest.approx(0.5)
    assert low == pytest.approx(0.478, abs=0.005)
    assert high == pytest.approx(0.522, abs=0.005)


def test_consistent_improvement_is_significant():
    a = np.linspace(0.0, 0.3, 500)
    b = a + 0.01                        # B beats A by exactly 0.01 for every user
    result = compare_paired(a, b)
    assert result["ci_low"] == pytest.approx(0.01) and result["ci_high"] == pytest.approx(0.01)
    assert result["p_value"] < 0.001
    assert result["share_better"] == 1.0


def test_no_real_difference_is_not_significant():
    a = np.full(2000, 0.2)
    b = a + np.array([0.1, -0.1] * 1000)   # half the users better, half worse, by the same amount
    result = compare_paired(a, b)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["p_value"] > 0.5