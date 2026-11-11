import re

import numpy as np
import pandas as pd

DAY = 24 * 3600
FEATURES = ["als_score", "als_rank", "genre_match", "likes_90d", "likes_365d",
            "movie_year", "user_likes_365d", "in_both_lists"]
YEAR_IN_TITLE = re.compile(r"\((\d{4})\)\s*$")   # "Inception (2010)" -> 2010


def movie_year(title: str) -> float:
    match = YEAR_IN_TITLE.search(title)
    return float(match.group(1)) if match else np.nan


def likes_per_movie(history: pd.DataFrame, start_ts: int, end_ts: int, threshold: float) -> pd.Series:
    """Number of likes each movie got between start_ts and end_ts."""
    window = history[(history["ts"] >= start_ts) & (history["ts"] < end_ts) & (history["rating"] >= threshold)]
    return window.groupby("movie_id").size()


def genre_match(cands: pd.DataFrame, past_likes: pd.DataFrame, movie_genres: dict) -> np.ndarray:
    """Average, over the candidate's genres, of the share of the user's past likes that include that genre."""
    genres = sorted({g for gs in movie_genres.values() for g in gs})
    g_col = {g: i for i, g in enumerate(genres)}
    m_row = {m: i for i, m in enumerate(movie_genres)}
    movie_matrix = np.zeros((len(m_row), len(genres)), dtype=np.float32)   # 1 where a movie has a genre
    for m, gs in movie_genres.items():
        for g in gs:
            movie_matrix[m_row[m], g_col[g]] = 1.0

    # Each user's genre profile: share of their likes that include each genre.
    likes = past_likes[past_likes["user_id"].isin(set(cands["user_id"])) & past_likes["movie_id"].isin(m_row)]
    rows = movie_matrix[likes["movie_id"].map(m_row).to_numpy()]
    profile = pd.DataFrame(rows, index=likes["user_id"].to_numpy()).groupby(level=0).mean()
    u_row = {u: i for i, u in enumerate(profile.index)}
    profile_matrix = profile.to_numpy()

    result = np.full(len(cands), np.nan, dtype=np.float32)
    cu, cm = cands["user_id"].map(u_row), cands["movie_id"].map(m_row)
    ok = (cu.notna() & cm.notna()).to_numpy()
    user_part = profile_matrix[cu[ok].astype(int)]
    movie_part = movie_matrix[cm[ok].astype(int)]
    n_genres = movie_part.sum(axis=1)
    result[ok] = np.where(n_genres > 0, (user_part * movie_part).sum(axis=1) / np.maximum(n_genres, 1), 0.0)
    return result


def add_features(cands: pd.DataFrame, history: pd.DataFrame, movies: pd.DataFrame,
                 movie_genres: dict, as_of_ts: int, threshold: float) -> pd.DataFrame:
    """Add the re-ranker features. Uses ONLY ratings before the pretend date."""
    assert history["ts"].max() < as_of_ts, "Leakage: history contains ratings on/after the pretend date"
    out = cands.copy()
    out["likes_90d"] = out["movie_id"].map(likes_per_movie(history, as_of_ts - 90 * DAY, as_of_ts, threshold)).fillna(0)
    out["likes_365d"] = out["movie_id"].map(likes_per_movie(history, as_of_ts - 365 * DAY, as_of_ts, threshold)).fillna(0)
    out["movie_year"] = out["movie_id"].map({m: movie_year(t) for m, t in zip(movies["movie_id"], movies["title"])})
    recent_likes = history[(history["ts"] >= as_of_ts - 365 * DAY) & (history["rating"] >= threshold)]
    out["user_likes_365d"] = out["user_id"].map(recent_likes.groupby("user_id").size()).fillna(0)
    out["in_both_lists"] = (out["from_als"] & out["from_pop"]).astype(int)
    out["genre_match"] = genre_match(out, history[history["rating"] >= threshold], movie_genres)
    return out


def add_labels(cands: pd.DataFrame, answers: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """label = 1 if the user liked the movie in the answer period, else 0."""
    liked = answers.loc[answers["rating"] >= threshold, ["user_id", "movie_id"]].assign(label=1)
    out = cands.merge(liked, on=["user_id", "movie_id"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    return out