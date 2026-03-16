from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import Settings
from app.db.repositories.recommendation_repository import RecommendationRepository
from app.features.context_features import resolve_meal_time
from app.features.user_features import build_user_profile
from app.ranking.filters import apply_filters
from app.ranking.rerank import rerank_items
from app.recommenders.hybrid import HybridRecommender
from app.schemas.request import FeedbackRequest, RecommendationRequest
from app.schemas.response import FeedbackResponse, HealthResponse, RecommendedItem, RecommendationResponse
from app.services.cache_service import CacheService


class RecommendationService:
    def __init__(
        self,
        repository: RecommendationRepository,
        settings: Settings,
        cache_service: CacheService,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.cache = cache_service
        self.hybrid = HybridRecommender()

    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        trace_id = str(uuid4())
        cache_key = self._build_cache_key(request)
        cached = self.cache.get(cache_key)
        if cached:
            cached.trace_id = trace_id
            return cached

        catalog = self.repository.load_catalog(self.settings.candidate_limit)
        filtered_catalog = apply_filters(catalog, request)
        history = self.repository.load_user_history(request.user_id)
        profile = build_user_profile(history, filtered_catalog)
        preferred_tokens = self._build_preferred_tokens(profile, request)
        resolved_meal_time = resolve_meal_time(request.meal_time)

        items: list[RecommendedItem] = []
        for food in filtered_catalog:
            score, reason = self.hybrid.score(
                food=food,
                profile=profile,
                preferred_tokens=preferred_tokens,
                request_tags=request.dietary_tags,
                resolved_meal_time=resolved_meal_time,
            )
            items.append(
                RecommendedItem(
                    food_id=food.food_id,
                    name=food.name,
                    category=food.category,
                    cuisine=food.cuisine,
                    price=food.price,
                    score=round(score, 4),
                    reason=reason,
                    metadata={
                        "meal_time": food.meal_time,
                        "tags": food.tags,
                        "available": food.available,
                    },
                )
            )

        strategy = "hybrid-personalized" if history else "hybrid-cold-start"
        response = RecommendationResponse(
            trace_id=trace_id,
            strategy=strategy,
            generated_at=datetime.now(timezone.utc),
            items=rerank_items(items, request.limit or self.settings.default_limit),
        )
        self.cache.set(cache_key, response.model_copy(deep=True))
        return response

    def healthcheck(self) -> HealthResponse:
        database_status = "up" if self.repository.ping() else "down"
        return HealthResponse(
            status="ok" if database_status == "up" else "degraded",
            database=database_status,
            schema=self.settings.resolved_db_schema,
        )

    def accept_feedback(self, feedback: FeedbackRequest) -> FeedbackResponse:
        return FeedbackResponse(accepted=True, trace_id=str(uuid4()))

    @staticmethod
    def _build_preferred_tokens(profile: dict[str, object], request: RecommendationRequest) -> set[str]:
        tokens = set()
        tokens.update(profile["preferred_categories"].keys())  # type: ignore[arg-type]
        tokens.update(profile["preferred_cuisines"].keys())  # type: ignore[arg-type]
        tokens.update(profile["preferred_tags"].keys())  # type: ignore[arg-type]
        if request.category:
            tokens.add(request.category.lower())
        if request.cuisine:
            tokens.add(request.cuisine.lower())
        if request.meal_time:
            tokens.add(request.meal_time.lower())
        return tokens

    @staticmethod
    def _build_cache_key(request: RecommendationRequest) -> str:
        return "|".join(
            [
                request.user_id or "anonymous",
                str(request.limit),
                request.category or "",
                request.cuisine or "",
                request.meal_time or "",
                ",".join(sorted(request.dietary_tags)),
                ",".join(sorted(request.exclude_ids)),
                request.location or "",
            ]
        )