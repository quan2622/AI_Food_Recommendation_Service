from app.db.repositories.recommendation_repository import FoodCandidate


class PopularRecommender:
    def score(self, food: FoodCandidate) -> float:
        return min(food.popularity_count / 20, 1.0)
