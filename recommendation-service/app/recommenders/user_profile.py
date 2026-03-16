from collections import Counter

from app.db.repositories.recommendation_repository import FoodRecord


class UserProfileRecommender:
    def score(self, food: FoodRecord, profile: dict[str, object]) -> float:
        category_scores: Counter[str] = profile["preferred_categories"]  # type: ignore[assignment]
        cuisine_scores: Counter[str] = profile["preferred_cuisines"]  # type: ignore[assignment]
        tag_scores: Counter[str] = profile["preferred_tags"]  # type: ignore[assignment]
        average_price = profile["average_price"]

        score = 0.0
        if food.category:
            score += category_scores.get(food.category.lower(), 0.0)
        if food.cuisine:
            score += cuisine_scores.get(food.cuisine.lower(), 0.0)
        for tag in food.tags:
            score += tag_scores.get(tag.lower(), 0.0)
        if average_price is not None and food.price is not None:
            delta = abs(food.price - float(average_price))
            score += max(0.0, 1 - (delta / max(float(average_price), 1.0)))
        return min(score / 5, 1.0)