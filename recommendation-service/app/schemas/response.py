from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    food_id: str
    name: str
    category: str | None = None
    cuisine: str | None = None
    price: float | None = None
    score: float
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    trace_id: str
    strategy: str
    generated_at: datetime
    items: list[RecommendedItem]


class HealthResponse(BaseModel):
    status: str
    database: str
    schema: str


class FeedbackResponse(BaseModel):
    accepted: bool
    trace_id: str