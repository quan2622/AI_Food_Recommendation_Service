from app.db.repositories.recommendation_repository import RecommendationRepository


def refresh_stats(repository: RecommendationRepository) -> dict[str, int]:
    catalog = repository.load_catalog(limit=500)
    summary = repository.summarize_catalog(catalog)
    return {
        "foods": len(catalog),
        "categories": len(summary["categories"]),
        "cuisines": len(summary["cuisines"]),
    }