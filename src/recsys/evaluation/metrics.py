import math
from itertools import combinations


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of the user's relevant movies that appear in the top k recommendations."""
    if not relevant:
        raise ValueError("User has no relevant movies; skip them before scoring.")
    hits = sum(1 for movie in recommended[:k] if movie in relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Points for hits in the top k. A hit at position p earns 1 / log2(p + 1)."""
    total = 0.0
    for position, movie in enumerate(recommended[:k], start=1):
        if movie in relevant:
            total += 1 / math.log2(position + 1)
    return total


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """DCG divided by the best possible DCG for this user, so it's between 0 and 1."""
    if not relevant:
        raise ValueError("User has no relevant movies; skip them before scoring.")
    # The perfect list puts relevant movies in the top positions,
    # but there are only k slots, so at most k of them can count.
    n_ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1 / math.log2(position + 1) for position in range(1, n_ideal_hits + 1))
    return dcg_at_k(recommended, relevant, k) / ideal_dcg

def catalog_coverage(all_recommendations: list[list], n_catalog: int, k: int) -> float:
    """Fraction of the catalog that appears in at least one user's top k."""
    shown = set()
    for recommended in all_recommendations:
        shown.update(recommended[:k])
    return len(shown) / n_catalog


def genre_difference(genres_a: set, genres_b: set) -> float:
    """1 - (shared genres / all genres). 0 = same genres, 1 = nothing in common."""
    all_genres = genres_a | genres_b
    if not all_genres:
        return 0.0  # neither movie has genre info, so we can't call them different
    shared = genres_a & genres_b
    return 1 - len(shared) / len(all_genres)


def intra_list_diversity(recommended: list, movie_genres: dict, k: int) -> float:
    """Average genre difference over every pair of movies in one user's top k."""
    top = recommended[:k]
    if len(top) < 2:
        raise ValueError("Need at least 2 movies to measure diversity.")
    pairs = list(combinations(top, 2))
    total = sum(
        genre_difference(movie_genres.get(a, set()), movie_genres.get(b, set()))
        for a, b in pairs
    )
    return total / len(pairs)