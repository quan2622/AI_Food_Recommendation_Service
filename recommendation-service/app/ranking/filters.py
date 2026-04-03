from app.db.repositories.recommendation_repository import FoodCandidate, UserContextRecord


def apply_filters(
    catalog: list[FoodCandidate],
    user_context: UserContextRecord,
    meal_type: str,
    meal_affinity_threshold: float,
) -> list[FoodCandidate]:
    allergy_set = {value.lower() for value in user_context.allergy_warnings}
    filtered: list[FoodCandidate] = []
    for food in catalog:
        if allergy_set.intersection({allergen.lower() for allergen in food.allergens}):
            continue
        if food.meal_affinity.get(meal_type, 0.25) < meal_affinity_threshold:
            continue
        if food.nutrition.calories <= 0:
            continue
        filtered.append(food)
    return filtered
