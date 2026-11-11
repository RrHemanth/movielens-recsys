import numpy as np


def bootstrap_ci(values, n_boot: int = 10_000, level: float = 0.95, seed: int = 42, chunk: int = 1_000):
    """Mean of `values` and a bootstrap confidence interval (resampling users with replacement)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_boot, chunk):          # in chunks, to keep memory small
        size = min(chunk, n_boot - start)
        picks = rng.integers(0, len(values), size=(size, len(values)))   # each row = one replay
        means.append(values[picks].mean(axis=1))
    means = np.concatenate(means)
    tail = (1 - level) / 2
    return float(values.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def sign_flip_pvalue(diffs, n_perm: int = 10_000, seed: int = 42, chunk: int = 1_000) -> float:
    """Two-sided p-value: how often do random sign flips give an average at least this far from 0?"""
    diffs = np.asarray(diffs, dtype=float)
    observed = abs(diffs.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, n_perm, chunk):
        size = min(chunk, n_perm - start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(diffs)))
        count += int((np.abs((signs * diffs).mean(axis=1)) >= observed).sum())
    return (count + 1) / (n_perm + 1)   # +1: never report exactly 0 from a finite simulation


def compare_paired(a, b, **kwargs) -> dict:
    """Compare model B with model A on the SAME users (per-user scores in the same order)."""
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    mean, low, high = bootstrap_ci(diffs, **kwargs)
    return {
        "mean_diff": mean, "ci_low": low, "ci_high": high,
        "p_value": sign_flip_pvalue(diffs),
        "share_better": float((diffs > 0).mean()),
        "share_worse": float((diffs < 0).mean()),
    }