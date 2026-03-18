from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


MealTypeLiteral = Literal["BREAKFAST", "LUNCH", "DINNER", "SNACK"]


class RecommendationRequest(BaseModel):
    user_id: int | None = None
    meal_type: MealTypeLiteral
    current_time: datetime | None = None
    limit: int = Field(default=10, ge=1, le=50)
    exclude_food_ids: list[int] = Field(default_factory=list)
    meal_affinity_threshold: float = Field(default=0.15, ge=0, le=1)

    @field_validator("meal_type", mode="before")
    @classmethod
    def normalize_meal_type(cls, value: str) -> str:
        return str(value).upper()


class FeedbackRequest(BaseModel):
    user_id: int | None = None
    food_id: int
    event_type: Literal["impression", "click", "favorite", "order", "rating"]
    rating: float | None = Field(default=None, ge=0, le=5)
    timestamp: datetime | None = None
