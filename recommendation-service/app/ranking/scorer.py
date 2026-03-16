def weighted_score(
    user_profile_score: float,
    content_score: float,
    popularity_score: float,
    context_score: float,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
) -> float:
    w1, w2, w3, w4 = weights
    return (
        w1 * user_profile_score
        + w2 * content_score
        + w3 * popularity_score
        + w4 * context_score
    )