from app.db.repositories.recommendation_repository import FoodRecord
from app.features.food_features import build_food_tokens


class ContentBasedRecommender:
    def score(self, food: FoodRecord, request_tags: list[str], preferred_tokens: set[str]) -> float:
        tokens = build_food_tokens(food)
        request_token_set = {tag.lower() for tag in request_tags if tag}
        overlap = len(tokens & preferred_tokens)
        request_overlap = len(tokens & request_token_set)
        denominator = max(len(preferred_tokens) + len(request_token_set), 1)
        return min((overlap + request_overlap) / denominator, 1.0)