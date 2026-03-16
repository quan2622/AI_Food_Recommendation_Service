from app.db.repositories.recommendation_repository import FoodRecord


def build_food_tokens(food: FoodRecord) -> set[str]:
    tokens: set[str] = set()
    for value in [food.category, food.cuisine, food.meal_time]:
        if value:
            tokens.add(value.strip().lower())
    for tag in food.tags:
        tokens.add(tag.lower())
    if food.description:
        tokens.update(word.lower() for word in food.description.split()[:20])
    return tokens