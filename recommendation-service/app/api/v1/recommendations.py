from datetime import datetime

from fastapi import APIRouter, Depends, Query, Response

from app.core.dependencies import get_recommendation_service
from app.schemas.request import RecommendationRequest
from app.schemas.response import RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/v1", tags=["recommendations"])


@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    response: Response,
    user_id: int | None = None,
    meal_type: str = Query(..., description="BREAKFAST | LUNCH | DINNER | SNACK"),
    current_time: datetime | None = None,
    limit: int = Query(default=10, ge=1, le=50),
    exclude_food_ids: list[int] = Query(default=[]),
    meal_affinity_threshold: float = Query(default=0.15, ge=0, le=1),
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    request = RecommendationRequest(
        user_id=user_id,
        meal_type=meal_type,
        current_time=current_time,
        limit=limit,
        exclude_food_ids=exclude_food_ids,
        meal_affinity_threshold=meal_affinity_threshold,
    )
    result = service.get_recommendations(request)
    if result.metadata.status_code != 200:
        response.status_code = result.metadata.status_code
    return result


@router.post("/recommendations/query", response_model=RecommendationResponse)
def query_recommendations(
    response: Response,
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    result = service.get_recommendations(payload)
    if result.metadata.status_code != 200:
        response.status_code = result.metadata.status_code
    return result
