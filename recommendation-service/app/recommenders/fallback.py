from app.db.repositories.recommendation_repository import FoodRecord


class FallbackRecommender:
    def score(self, food: FoodRecord) -> float:
        return 0.2 if food.available else 0.0