from app.db.repositories.recommendation_repository import FoodCandidate


class FallbackRecommender:
    def score(self, food: FoodCandidate) -> float:
        return 0.1 if food.nutrition.calories > 0 else 0.0
