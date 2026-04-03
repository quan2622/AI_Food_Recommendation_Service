from collections import Counter

from app.db.repositories.recommendation_repository import FoodCandidate


def build_user_profile(history_frequencies: dict[int, float], catalog: list[FoodCandidate]) -> dict[str, object]:
    foods_by_id = {food.food_id: food for food in catalog}
    category_scores: Counter[str] = Counter()

    for food_id, weight in history_frequencies.items():
        food = foods_by_id.get(food_id)
        if not food or not food.category.name:
            continue
        category_scores[food.category.name.lower()] += weight

    return {
        "preferred_categories": category_scores,
        "history_food_ids": set(history_frequencies.keys()),
    }
