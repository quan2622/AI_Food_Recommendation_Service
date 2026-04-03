from fastapi import APIRouter, Depends

from app.core.dependencies import get_recommendation_service
from app.schemas.response import HealthResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck(service: RecommendationService = Depends(get_recommendation_service)) -> HealthResponse:
    return service.healthcheck()