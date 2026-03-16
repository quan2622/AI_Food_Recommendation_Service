from app.db.repositories.recommendation_repository import FoodRecord


class PopularRecommender:
    def score(self, food: FoodRecord) -> float:
        popularity = min(food.popularity / 100 if food.popularity else 0.0, 1.0)
        rating = min(food.rating / 5 if food.rating else 0.0, 1.0)
        return popularity * 0.7 + rating * 0.3