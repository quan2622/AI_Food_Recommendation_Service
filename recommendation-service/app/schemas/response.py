from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Pagination(BaseModel):
    total_items: int
    current_page: int = 1
    total_pages: int = 1


class ResponseMetadata(BaseModel):
    status_code: int = Field(alias="statusCode")
    message: str
    ec: int = Field(alias="EC")
    timestamp: datetime
    pagination: Pagination | None = None

    model_config = ConfigDict(populate_by_name=True)


class CategorySummary(BaseModel):
    id: int | None = None
    name: str | None = None


class RecommendationContext(BaseModel):
    score: float
    reason: str
    tags: list[str] = Field(default_factory=list)


class Macronutrients(BaseModel):
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0


class NutritionSummary(BaseModel):
    calories: float = 0.0
    macronutrients: Macronutrients
    suggested_portion_grams: float = 0.0


class GoalAlignment(BaseModel):
    calories: str = "Normal"
    protein: str = "Normal"
    fat: str = "Normal"
    fiber: str = "Normal"


class HealthAnalysis(BaseModel):
    is_safe: bool
    allergens_detected: list[str] = Field(default_factory=list)
    goal_alignment: GoalAlignment


class RecommendedItem(BaseModel):
    id: int
    food_name: str = Field(alias="foodName")
    description: str | None = None
    image_url: str | None = Field(default=None, alias="imageUrl")
    category: CategorySummary
    recommendation_context: RecommendationContext
    nutrition: NutritionSummary
    health_analysis: HealthAnalysis

    model_config = ConfigDict(populate_by_name=True)


class UserContextPayload(BaseModel):
    calories_remaining: float = 0.0
    burned_calories_today: float = 0.0
    allergy_warnings: list[str] = Field(default_factory=list)


class UserSummaryPayload(BaseModel):
    id: int | None = None
    name: str | None = None


class RecommendationData(BaseModel):
    recommendation_strategy: str
    user: UserSummaryPayload
    user_context: UserContextPayload
    items: list[RecommendedItem]


class RecommendationResponse(BaseModel):
    metadata: ResponseMetadata
    data: RecommendationData | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    schema: str


class FeedbackResponse(BaseModel):
    accepted: bool
    trace_id: str

