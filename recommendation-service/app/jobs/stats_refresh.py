from app.db.repositories.recommendation_repository import RecommendationRepository


def refresh_stats(repository: RecommendationRepository) -> dict[str, int]:
    summary = repository.summarize_food_catalog(limit=500)
    return {
        "foods": int(summary.get("foods", 0)),
        "categories": int(summary.get("categories", 0)),
        "foods_with_nutrition": int(summary.get("foods_with_nutrition", 0)),
        "new_items_7d": int(summary.get("new_items_7d", 0)),
    }
