from datetime import datetime, timezone

from app.db.repositories.recommendation_repository import CategoryRecord, FoodCandidate, NutritionVector, UserContextRecord
from app.recommenders.hybrid import HybridRecommender


def test_hybrid_penalizes_repeated_food_and_prefers_matching_nutrition():
    recommender = HybridRecommender()
    context = UserContextRecord(
        user_id=1,
        goal_type="WEIGHT_GAIN",
        remaining_nutrition=NutritionVector(calories=600, protein=45, carbs=50, fat=15, fiber=8),
        repeat_counts={1: 3},
        total_logs=5,
    )
    strong_match = FoodCandidate(
        food_id=1,
        food_name="Com ga",
        created_at=datetime.now(timezone.utc),
        category=CategoryRecord(category_id=1, name="Com"),
        nutrition=NutritionVector(calories=320, protein=28, carbs=35, fat=8, fiber=3),
        meal_affinity={"LUNCH": 0.8, "BREAKFAST": 0.1, "DINNER": 0.1, "SNACK": 0.1},
    )
    weak_match = FoodCandidate(
        food_id=2,
        food_name="Banh ngot",
        created_at=datetime.now(timezone.utc),
        category=CategoryRecord(category_id=2, name="Dessert"),
        nutrition=NutritionVector(calories=120, protein=2, carbs=18, fat=6, fiber=1),
        meal_affinity={"LUNCH": 0.3, "BREAKFAST": 0.4, "DINNER": 0.1, "SNACK": 0.4},
    )

    strong_score = recommender.score(strong_match, context, "LUNCH", collaborative_score=0.2)
    weak_score = recommender.score(weak_match, context, "LUNCH", collaborative_score=0.0)

    assert strong_score.content_score > weak_score.content_score
    assert strong_score.repeat_penalty > 0
    assert strong_score.collaborative_score > weak_score.collaborative_score
    assert strong_score.final_score > weak_score.final_score
