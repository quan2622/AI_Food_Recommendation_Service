from app.db.repositories.recommendation_repository import FoodRecord
from app.ranking.scorer import weighted_score
from app.recommenders.content_based import ContentBasedRecommender
from app.recommenders.fallback import FallbackRecommender
from app.recommenders.popular import PopularRecommender
from app.recommenders.user_profile import UserProfileRecommender


class HybridRecommender:
    def __init__(self) -> None:
        self.popular = PopularRecommender()
        self.content = ContentBasedRecommender()
        self.user_profile = UserProfileRecommender()
        self.fallback = FallbackRecommender()

    def score(
        self,
        food: FoodRecord,
        profile: dict[str, object],
        preferred_tokens: set[str],
        request_tags: list[str],
        resolved_meal_time: str,
    ) -> tuple[float, str]:
        user_score = self.user_profile.score(food, profile) if profile["history_food_ids"] else 0.0
        content_score = self.content.score(food, request_tags, preferred_tokens)
        popularity_score = self.popular.score(food)
        context_score = 1.0 if food.meal_time and food.meal_time.lower() == resolved_meal_time.lower() else 0.2
        if not profile["history_food_ids"]:
            context_score = max(context_score, self.fallback.score(food))

        final_score = weighted_score(user_score, content_score, popularity_score, context_score)
        if user_score > 0.45:
            reason = "Matched your taste profile"
        elif content_score > 0.35:
            reason = "Relevant to your requested filters"
        elif popularity_score > 0.2:
            reason = "Trending among users"
        else:
            reason = "Fallback recommendation"
        return final_score, reason