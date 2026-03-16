from collections import Counter

from app.db.repositories.recommendation_repository import FoodRecord, InteractionRecord


def build_user_profile(history: list[InteractionRecord], catalog: list[FoodRecord]) -> dict[str, object]:
    foods_by_id = {food.food_id: food for food in catalog}
    category_scores: Counter[str] = Counter()
    cuisine_scores: Counter[str] = Counter()
    tag_scores: Counter[str] = Counter()
    price_values: list[float] = []

    for interaction in history:
        food = foods_by_id.get(interaction.food_id)
        if not food:
            continue
        if food.category:
            category_scores[food.category.lower()] += interaction.weight
        if food.cuisine:
            cuisine_scores[food.cuisine.lower()] += interaction.weight
        for tag in food.tags:
            tag_scores[tag.lower()] += interaction.weight
        if food.price is not None:
            price_values.append(food.price)

    return {
        "preferred_categories": category_scores,
        "preferred_cuisines": cuisine_scores,
        "preferred_tags": tag_scores,
        "average_price": (sum(price_values) / len(price_values)) if price_values else None,
        "history_food_ids": {interaction.food_id for interaction in history},
    }