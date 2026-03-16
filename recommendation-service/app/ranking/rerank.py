from app.schemas.response import RecommendedItem


def rerank_items(items: list[RecommendedItem], limit: int) -> list[RecommendedItem]:
    return sorted(items, key=lambda item: item.score, reverse=True)[:limit]