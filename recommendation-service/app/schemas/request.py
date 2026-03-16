from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    limit: int = Field(default=10, ge=1, le=50)
    category: str | None = None
    cuisine: str | None = None
    meal_time: Literal["breakfast", "lunch", "dinner", "snack"] | None = None
    dietary_tags: list[str] = Field(default_factory=list)
    exclude_ids: list[str] = Field(default_factory=list)
    location: str | None = None


class FeedbackRequest(BaseModel):
    user_id: str | None = None
    food_id: str
    event_type: Literal["impression", "click", "favorite", "order", "rating"]
    rating: float | None = Field(default=None, ge=0, le=5)
    timestamp: datetime | None = None