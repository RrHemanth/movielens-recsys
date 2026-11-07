import math


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