from fastapi.testclient import TestClient

from app.core.dependencies import get_recommendation_service
from app.main import app
from app.schemas.request import FeedbackRequest, RecommendationRequest
from app.schemas.response import (
    CategorySummary,
    FeedbackResponse,
    GoalAlignment,
    HealthAnalysis,
    HealthResponse,
    Macronutrients,
    NutritionSummary,
    Pagination,
    RecommendationContext,
    RecommendationData,
    RecommendationResponse,
    RecommendedItem,
    ResponseMetadata,
    UserContextPayload,
    UserSummaryPayload,
)


class StubService:
    def healthcheck(self) -> HealthResponse:
        return HealthResponse(status="ok", database="up", schema="public")

    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        return RecommendationResponse(
            metadata=ResponseMetadata(
                statusCode=200,
                message="ok",
                EC=0,
                timestamp="2026-03-18T00:00:00Z",
                pagination=Pagination(total_items=1),
            ),
            data=RecommendationData(
                recommendation_strategy="content-based-filtering",
                user=UserSummaryPayload(id=1, name="Test User"),
                user_context=UserContextPayload(calories_remaining=500, burned_calories_today=0, allergy_warnings=[]),
                items=[
                    RecommendedItem(
                        id=1,
                        foodName="Pho",
                        description="test",
                        imageUrl=None,
                        category=CategorySummary(id=10, name="Mon nuoc"),
                        recommendation_context=RecommendationContext(
                            score=0.88,
                            reason="Phu hop nhu cau dinh duong",
                            tags=["High Protein"],
                        ),
                        nutrition=NutritionSummary(
                            calories=450,
                            macronutrients=Macronutrients(protein=30, carbs=40, fat=10, fiber=2),
                            suggested_portion_grams=200,
                        ),
                        health_analysis=HealthAnalysis(
                            is_safe=True,
                            allergens_detected=[],
                            goal_alignment=GoalAlignment(calories="Optimized", protein="Excellent", fat="Good", fiber="Normal"),
                        ),
                    )
                ],
            ),
        )

    def accept_feedback(self, feedback: FeedbackRequest) -> FeedbackResponse:
        return FeedbackResponse(accepted=True, trace_id="trace-2")


app.dependency_overrides[get_recommendation_service] = lambda: StubService()
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_recommendations_endpoint():
    response = client.get("/v1/recommendations", params={"limit": 1, "meal_type": "MEAL_LUNCH"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["data"]["recommendation_strategy"] == "content-based-filtering"
    assert payload["data"]["user"]["id"] == 1
    assert payload["data"]["user"]["name"] == "Test User"
    assert len(payload["data"]["items"]) == 1
    assert payload["data"]["items"][0]["foodName"] == "Pho"


def test_feedback_endpoint():
    response = client.post(
        "/v1/feedback",
        json={"food_id": 1, "event_type": "click", "user_id": 1},
    )
    assert response.status_code == 202
    assert response.json()["accepted"] is True

