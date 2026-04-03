from app.db.repositories.recommendation_repository import FoodCandidate, UserContextRecord
from app.recommenders.hybrid import HybridRecommender


class ContentBasedRecommender:
    def __init__(self) -> None:
        self.hybrid = HybridRecommender()

    def score(self, food: FoodCandidate, context: UserContextRecord, meal_type: str) -> float:
        return self.hybrid._content_score(food, context, meal_type)
