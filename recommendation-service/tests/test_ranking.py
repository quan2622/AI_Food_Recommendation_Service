from app.db.repositories.recommendation_repository import FoodRecord, InteractionRecord
from app.features.user_features import build_user_profile
from app.recommenders.hybrid import HybridRecommender


def test_hybrid_scores_user_preference_higher():
    catalog = [
        FoodRecord(food_id="1", name="Pho Bo", category="Noodle", cuisine="Vietnamese", tags=["beef"], popularity=50),
        FoodRecord(food_id="2", name="Salad", category="Healthy", cuisine="Western", tags=["vegan"], popularity=50),
    ]
    history = [InteractionRecord(food_id="1", event_type="order", weight=1.0)]
    profile = build_user_profile(history, catalog)

    recommender = HybridRecommender()
    top_score, _ = recommender.score(catalog[0], profile, {"noodle", "vietnamese"}, [], "lunch")
    lower_score, _ = recommender.score(catalog[1], profile, {"noodle", "vietnamese"}, [], "lunch")

    assert top_score > lower_score