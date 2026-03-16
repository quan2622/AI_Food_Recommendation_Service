from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_recommendation_service
from app.schemas.request import FeedbackRequest
from app.schemas.response import FeedbackResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_feedback(
    payload: FeedbackRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> FeedbackResponse:
    return service.accept_feedback(payload)