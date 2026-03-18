from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.core.config import Settings
from app.db.repositories.recommendation_repository import FoodCandidate, RecommendationRepository, UserContextRecord
from app.recommenders.hybrid import HybridRecommender, ScoreBreakdown
from app.schemas.request import FeedbackRequest, RecommendationRequest
from app.schemas.response import (
    CategorySummary,
    FeedbackResponse,
    GoalAlignment,
    HealthAnalysis,
    HealthResponse,
    Macronutrients,
    NutritionSummary,
    Pagination,
    RecommendationContext,
    RecommendationData,
    RecommendationResponse,
    RecommendedItem,
    ResponseMetadata,
    UserContextPayload,
)
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
        current_time = self._ensure_aware(request.current_time or datetime.now(timezone.utc))
        user_context = self.repository.load_user_context(request.user_id, current_time)
        candidates = self.repository.load_food_candidates(self.settings.candidate_limit)
        candidates = [food for food in candidates if food.food_id not in set(request.exclude_food_ids)]
        filtered = self._apply_hard_filters(
            candidates,
            user_context,
            request.meal_type,
            request.meal_affinity_threshold,
        )

        if len(filtered) < 5:
            filtered = self._apply_hard_filters(
                candidates,
                user_context,
                request.meal_type,
                request.meal_affinity_threshold / 2,
            )
        if not filtered:
            filtered = candidates

        collaborative_scores = self._load_collaborative_scores(request.user_id, filtered)
        scored: list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]] = []
        for food in filtered:
            breakdown = self.hybrid.score(
                food,
                user_context,
                request.meal_type,
                collaborative_score=collaborative_scores.get(food.food_id, 0.0),
                repeat_threshold=self.settings.repeat_penalty_threshold,
            )
            reason, tags = self._build_reason_and_tags(food, user_context, breakdown, current_time)
            scored.append((food, breakdown, reason, tags))

        strategy = self._resolve_strategy(scored)
        if strategy == "popular-fallback":
            ranked = self._popular_fallback(filtered, request.limit)
        else:
            ranked = self._rerank_with_diversity(scored, request.limit, current_time)

        items = [self._build_item(food, breakdown, reason, tags, user_context) for food, breakdown, reason, tags in ranked]
        return RecommendationResponse(
            metadata=ResponseMetadata(
                statusCode=200,
                message="Goi y mon an thanh cong",
                EC=0,
                timestamp=current_time,
                pagination=Pagination(total_items=len(items), current_page=1, total_pages=1),
            ),
            data=RecommendationData(
                recommendation_strategy=strategy,
                user_context=UserContextPayload(
                    calories_remaining=round(user_context.remaining_nutrition.calories, 2),
                    burned_calories_today=round(user_context.burned_calories_today, 2),
                    allergy_warnings=user_context.allergy_warnings,
                ),
                items=items,
            ),
        )

    def healthcheck(self) -> HealthResponse:
        database_status = "up" if self.repository.ping() else "down"
        return HealthResponse(
            status="ok" if database_status == "up" else "degraded",
            database=database_status,
            schema=self.settings.resolved_db_schema,
        )

    def accept_feedback(self, feedback: FeedbackRequest) -> FeedbackResponse:
        return FeedbackResponse(accepted=True, trace_id=str(uuid4()))

    def _load_collaborative_scores(self, user_id: int | None, candidates: list[FoodCandidate]) -> dict[int, float]:
        if user_id is None or not candidates:
            return {}
        cache_key = self._cf_cache_key(user_id, [food.food_id for food in candidates])
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            return cached
        scores = self.repository.load_candidate_collaborative_scores(
            user_id=user_id,
            candidate_ids=[food.food_id for food in candidates],
            neighbor_limit=self.settings.collaborative_neighbors,
            min_shared_items=self.settings.collaborative_min_shared_items,
            history_limit=self.settings.collaborative_history_limit,
        )
        self.cache.set(cache_key, scores)
        return scores

    def _resolve_strategy(self, scored: list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]]) -> str:
        if not scored or all(item[1].final_score <= 0 for item in scored):
            return "popular-fallback"
        if any(item[1].collaborative_score > 0 for item in scored):
            return "hybrid"
        return "content-based-filtering"

    def _apply_hard_filters(
        self,
        catalog: list[FoodCandidate],
        user_context: UserContextRecord,
        meal_type: str,
        threshold: float,
    ) -> list[FoodCandidate]:
        allergy_set = {value.lower() for value in user_context.allergy_warnings}
        filtered: list[FoodCandidate] = []
        for food in catalog:
            detected = {value.lower() for value in food.allergens}
            if allergy_set and allergy_set.intersection(detected):
                continue
            if food.meal_affinity.get(meal_type, 0.25) < threshold:
                continue
            if food.nutrition.calories <= 0:
                continue
            filtered.append(food)
        return filtered

    def _rerank_with_diversity(
        self,
        scored: list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]],
        limit: int,
        current_time: datetime,
    ) -> list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]]:
        sorted_items = sorted(scored, key=lambda item: item[1].final_score, reverse=True)
        limited: list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]] = []
        category_counts: Counter[str] = Counter()
        overflow: list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]] = []

        for item in sorted_items:
            category_name = item[0].category.name or f"uncategorized-{item[0].food_id}"
            if category_counts[category_name] >= 2 and len(limited) < 10:
                overflow.append(item)
                continue
            limited.append(item)
            category_counts[category_name] += 1
            if len(limited) == limit:
                break

        for item in overflow:
            if len(limited) == limit:
                break
            limited.append(item)

        new_item_index = min(2, max(len(limited) - 1, 0))
        new_item = next(
            (
                item
                for item in sorted_items
                if item[0].created_at
                and self._ensure_aware(item[0].created_at) >= current_time - timedelta(days=self.settings.new_item_window_days)
                and item[1].content_score >= 0.15
                and item not in limited[:new_item_index + 1]
            ),
            None,
        )
        if new_item and limited:
            limited = [item for item in limited if item[0].food_id != new_item[0].food_id]
            limited.insert(new_item_index, new_item)
            limited = limited[:limit]
        return limited

    @staticmethod
    def _popular_fallback(catalog: list[FoodCandidate], limit: int) -> list[tuple[FoodCandidate, ScoreBreakdown, str, list[str]]]:
        sorted_items = sorted(
            catalog,
            key=lambda food: (food.popularity_count, -food.nutrition.calories),
            reverse=True,
        )
        return [
            (
                food,
                ScoreBreakdown(
                    content_score=0.0,
                    collaborative_score=0.0,
                    repeat_penalty=0.0,
                    final_score=0.1,
                    meal_affinity=food.meal_affinity.get("LUNCH", 0.25),
                    alpha=0.0,
                    beta=0.0,
                    gamma=0.0,
                ),
                "Mon duoc nhieu nguoi chon vao buoi an nay",
                ["Popular Fallback"],
            )
            for food in sorted_items[:limit]
        ]

    def _build_item(
        self,
        food: FoodCandidate,
        breakdown: ScoreBreakdown,
        reason: str,
        tags: list[str],
        user_context: UserContextRecord,
    ) -> RecommendedItem:
        detected_allergens = [allergen for allergen in food.allergens if allergen in user_context.allergy_warnings]
        return RecommendedItem(
            id=food.food_id,
            foodName=food.food_name,
            description=food.description,
            imageUrl=food.image_url,
            category=CategorySummary(id=food.category.category_id, name=food.category.name),
            recommendation_context=RecommendationContext(
                score=round(breakdown.final_score, 4),
                reason=reason,
                tags=tags,
            ),
            nutrition=NutritionSummary(
                calories=round(food.nutrition.calories, 2),
                macronutrients=Macronutrients(
                    protein=round(food.nutrition.protein, 2),
                    carbs=round(food.nutrition.carbs, 2),
                    fat=round(food.nutrition.fat, 2),
                    fiber=round(food.nutrition.fiber, 2),
                ),
                suggested_portion_grams=round(self._suggested_portion_grams(food, user_context), 2),
            ),
            health_analysis=HealthAnalysis(
                is_safe=not detected_allergens,
                allergens_detected=detected_allergens,
                goal_alignment=self._goal_alignment(food, user_context),
            ),
        )

    @staticmethod
    def _suggested_portion_grams(food: FoodCandidate, user_context: UserContextRecord) -> float:
        calories = food.nutrition.calories
        if calories <= 0:
            return 100.0
        remaining_calories = user_context.remaining_nutrition.calories
        if remaining_calories <= 0:
            return 100.0
        return max(50.0, min((remaining_calories / calories) * 100, 500.0))

    @staticmethod
    def _goal_alignment(food: FoodCandidate, user_context: UserContextRecord) -> GoalAlignment:
        return GoalAlignment(
            calories=RecommendationService._alignment_label(food.nutrition.calories, user_context.remaining_nutrition.calories),
            protein=RecommendationService._alignment_label(food.nutrition.protein, user_context.remaining_nutrition.protein),
            fat=RecommendationService._alignment_label(food.nutrition.fat, user_context.remaining_nutrition.fat),
            fiber=RecommendationService._alignment_label(food.nutrition.fiber, user_context.remaining_nutrition.fiber),
        )

    @staticmethod
    def _alignment_label(value: float, remaining: float) -> str:
        if remaining <= 0:
            return "Normal"
        ratio = value / remaining
        if ratio >= 0.9:
            return "Excellent"
        if ratio >= 0.7:
            return "Optimized"
        if ratio >= 0.5:
            return "Good"
        return "Normal"

    @staticmethod
    def _build_reason_and_tags(
        food: FoodCandidate,
        user_context: UserContextRecord,
        breakdown: ScoreBreakdown,
        current_time: datetime,
    ) -> tuple[str, list[str]]:
        tags: list[str] = []
        if food.nutrition.protein >= max(user_context.remaining_nutrition.protein * 0.5, 20):
            tags.append("High Protein")
        if food.nutrition.fiber >= max(user_context.remaining_nutrition.fiber * 0.5, 5):
            tags.append("High Fiber")
        if breakdown.repeat_penalty == 0:
            tags.append("New Rotation")
        if food.created_at and RecommendationService._ensure_aware(food.created_at) >= current_time - timedelta(days=7):
            tags.append("New Item")
        if breakdown.collaborative_score > 0:
            tags.append("Community Match")

        if breakdown.content_score >= 0.65 and breakdown.collaborative_score >= 0.35:
            reason = "Vua phu hop dinh duong con lai, vua duoc nhung nguoi co thoi quen tuong tu chon"
        elif breakdown.content_score >= 0.65 and breakdown.repeat_penalty == 0:
            reason = "Vua phu hop nhu cau dinh duong con lai, vua la lua chon moi de tranh lap mon"
        elif breakdown.content_score >= 0.65:
            reason = "Phu hop voi luong dinh duong ban con thieu trong ngay"
        elif breakdown.collaborative_score >= 0.35:
            reason = "Nhung nguoi co muc tieu hoac lich su tuong tu thuong chon mon nay"
        elif breakdown.meal_affinity >= 0.35:
            reason = "Thuong duoc chon vao dung buoi an nay"
        else:
            reason = "Mon can bang cho muc tieu dinh duong hien tai"
        return reason, tags

    @staticmethod
    def _ensure_aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _cf_cache_key(user_id: int, candidate_ids: list[int]) -> str:
        suffix = ",".join(str(item_id) for item_id in sorted(candidate_ids)[:50])
        return f"cf:{user_id}:{suffix}"
