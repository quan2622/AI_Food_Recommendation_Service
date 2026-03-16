from fastapi.testclient import TestClient

from app.core.dependencies import get_recommendation_service
from app.main import app
from app.schemas.request import FeedbackRequest, RecommendationRequest
from app.schemas.response import FeedbackResponse, HealthResponse, RecommendationResponse, RecommendedItem


class StubService:
    def healthcheck(self) -> HealthResponse:
        return HealthResponse(status="ok", database="up", schema="public")

    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        return RecommendationResponse(
            trace_id="trace-1",
            strategy="hybrid-cold-start",
            generated_at="2026-03-16T00:00:00Z",
            items=[
                RecommendedItem(
                    food_id="1",
                    name="Pho",
                    category="Noodle",
                    cuisine="Vietnamese",
                    price=45000,
                    score=0.88,
                    reason="Trending among users",
                )
            ],
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
    response = client.get("/v1/recommendations", params={"limit": 1})
    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "hybrid-cold-start"
    assert len(payload["items"]) == 1


def test_feedback_endpoint():
    response = client.post(
        "/v1/feedback",
        json={"food_id": "1", "event_type": "click", "user_id": "u-1"},
    )
    assert response.status_code == 202
    assert response.json()["accepted"] is True