from fastapi import APIRouter, Depends, Query

from app.core.dependencies import get_recommendation_service
from app.schemas.request import RecommendationRequest
from app.schemas.response import RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/v1", tags=["recommendations"])


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str | None = None,
    limit: int = Query(default=10, ge=1, le=50),
    category: str | None = None,
    cuisine: str | None = None,
    meal_time: str | None = None,
    dietary_tags: list[str] = Query(default=[]),
    exclude_ids: list[str] = Query(default=[]),
    location: str | None = None,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    request = RecommendationRequest(
        user_id=user_id,
        limit=limit,
        category=category,
        cuisine=cuisine,
        meal_time=meal_time,
        dietary_tags=dietary_tags,
        exclude_ids=exclude_ids,
        location=location,
    )
    return service.get_recommendations(request)


@router.post("/recommendations/query", response_model=RecommendationResponse)
def query_recommendations(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    return service.get_recommendations(payload)