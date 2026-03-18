from app.db.repositories.recommendation_repository import FoodCandidate


def build_food_tokens(food: FoodCandidate) -> set[str]:
    tokens: set[str] = set()
    if food.category.name:
        tokens.add(food.category.name.lower())
    if food.description:
        tokens.update(word.lower() for word in food.description.split()[:20])
    for allergen in food.allergens:
        tokens.add(allergen.lower())
    return tokens
