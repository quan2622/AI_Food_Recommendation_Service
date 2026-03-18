from collections import Counter

from app.db.repositories.recommendation_repository import FoodCandidate


class UserProfileRecommender:
    def score(self, food: FoodCandidate, profile: dict[str, object]) -> float:
        category_scores: Counter[str] = profile.get("preferred_categories", Counter())  # type: ignore[assignment]
        if food.category.name:
            return min(category_scores.get(food.category.name.lower(), 0.0) / 5, 1.0)
        return 0.0
