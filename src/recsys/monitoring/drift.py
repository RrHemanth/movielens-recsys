import numpy as np
import pandas as pd

STABLE, WATCH = 0.1, 0.25   # PSI rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 drift


def psi_from_shares(ref_shares, cur_shares, eps: float = 1e-4) -> float:
    """PSI = sum over buckets of (current share - reference share) * ln(current / reference)."""
    ref = np.clip(np.asarray(ref_shares, dtype=float), eps, None)   # avoid log(0) for empty buckets
    cur = np.clip(np.asarray(cur_shares, dtype=float), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference, current, n_bins: int = 10) -> float:
    """PSI for numbers: buckets are cut at the reference's quantiles (about 10% of reference in each)."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])   # first and last bucket are open-ended
    ref_counts, _ = np.histogram(reference, edges)
    cur_counts, _ = np.histogram(current, edges)
    return psi_from_shares(ref_counts / ref_counts.sum(), cur_counts / cur_counts.sum())


def psi_categorical(reference, current) -> float:
    """PSI for categories (e.g. star values): each category is its own bucket."""
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return psi_from_shares(ref.reindex(categories, fill_value=0), cur.reindex(categories, fill_value=0))


def ks_statistic(reference, current) -> float:
    """Biggest gap between the two cumulative distributions (0 = identical, 1 = no overlap at all)."""
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    points = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, points, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, points, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def psi_status(value: float) -> str:
    return "stable" if value < STABLE else "watch" if value < WATCH else "drift"


def window_signals(ratings: pd.DataFrame, movie_year: dict, known_movies: set) -> dict:
    """The raw values of each signal for one window of ratings."""
    when = pd.to_datetime(ratings["ts"], unit="s")
    return {
        "activity": ratings.groupby([ratings["user_id"], when.dt.to_period("M")]).size().to_numpy(),
        "movie_age": (when.dt.year - ratings["movie_id"].map(movie_year)).dropna().to_numpy(),
        "rating": ratings["rating"].to_numpy(),
        "unknown_movie": (~ratings["movie_id"].isin(known_movies)).to_numpy(),
    }


def drift_report(reference: dict, current: dict) -> dict:
    """Compare one window's signals against the reference."""
    return {
        "psi_activity": psi_numeric(reference["activity"], current["activity"]),
        "psi_movie_age": psi_numeric(reference["movie_age"], current["movie_age"]),
        "psi_rating": psi_categorical(reference["rating"], current["rating"]),
        "psi_unknown_movie": psi_categorical(reference["unknown_movie"], current["unknown_movie"]),
        "ks_activity": ks_statistic(reference["activity"], current["activity"]),
        "ks_movie_age": ks_statistic(reference["movie_age"], current["movie_age"]),
        "unknown_movie_share": float(np.mean(current["unknown_movie"])),
    }