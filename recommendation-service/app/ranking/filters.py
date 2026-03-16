from app.db.repositories.recommendation_repository import FoodRecord
from app.schemas.request import RecommendationRequest


def apply_filters(catalog: list[FoodRecord], request: RecommendationRequest) -> list[FoodRecord]:
    excluded = set(request.exclude_ids)
    dietary_tags = {tag.lower() for tag in request.dietary_tags}
    filtered: list[FoodRecord] = []

    for food in catalog:
        if food.food_id in excluded:
            continue
        if not food.available:
            continue
        if request.category and (food.category or "").lower() != request.category.lower():
            continue
        if request.cuisine and (food.cuisine or "").lower() != request.cuisine.lower():
            continue
        if dietary_tags and not dietary_tags.issubset({tag.lower() for tag in food.tags}):
            continue
        filtered.append(food)
    return filtered