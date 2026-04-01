from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.db.repositories.recommendation_repository import FoodCandidate, UserContextRecord


@dataclass
class ScoreBreakdown:
    content_score: float
    collaborative_score: float
    repeat_penalty: float
    final_score: float
    meal_affinity: float
    alpha: float
    beta: float
    gamma: float


class HybridRecommender:
    def score(
        self,
        food: FoodCandidate,
        context: UserContextRecord,
        meal_type: str,
        collaborative_score: float = 0.0,
        repeat_threshold: int = 3,
        nutrition_priority: str = "BALANCED",
    ) -> ScoreBreakdown:
        content_score = self._content_score(food, context, meal_type, nutrition_priority)
        repeat_penalty = self._repeat_penalty(food.food_id, context, repeat_threshold)
        
        alpha, beta, gamma = self._dynamic_weights(
            context.total_logs, context.goal_type, collaborative_score > 0, nutrition_priority
        )
        
        # Hybrid score cơ bản
        hybrid_score = alpha * content_score + beta * collaborative_score - gamma * repeat_penalty
        hybrid_score = max(0.0, min(hybrid_score, 1.0))

        # === BOOST MẠNH SAU HYBRID (đây là phần fix quyết định) ===
        if nutrition_priority != "BALANCED":
            multiplier = self._nutrition_priority_multiplier(nutrition_priority, food)
            # ^0.65 giúp boost mạnh nhưng vẫn mượt, không làm score > 1.0
            final_score = hybrid_score * (multiplier ** 0.65)
        else:
            final_score = hybrid_score

        final_score = max(0.0, min(final_score, 1.0))

        return ScoreBreakdown(
            content_score=content_score,
            collaborative_score=collaborative_score,
            repeat_penalty=repeat_penalty,
            final_score=final_score,
            meal_affinity=food.meal_affinity.get(meal_type, 0.25),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def _content_score(self, food: FoodCandidate, context: UserContextRecord, meal_type: str, nutrition_priority: str = "BALANCED") -> float:
        remaining = context.remaining_nutrition.as_list()
        nutrition = food.nutrition.as_list()
        cosine = self._cosine_similarity(remaining, nutrition)
        
        if cosine == 0.0 and sum(remaining) == 0:
            base_score = 0.3
        else:
            base_score = cosine
            
        goal_multiplier = self._goal_multiplier(context.goal_type, food)
        meal_affinity = food.meal_affinity.get(meal_type, 0.25)
        # Không nhân nutrition_priority_multiplier ở đây vì đã có post-hybrid boost trong score()
        return max(0.0, min(base_score * goal_multiplier * meal_affinity, 1.0))

    @staticmethod
    def _repeat_penalty(food_id: int, context: UserContextRecord, threshold: int = 3) -> float:
        count = context.repeat_counts.get(food_id, 0)
        return max(0.0, min(count / threshold, 1.0))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        return max(0.0, min(dot / (left_norm * right_norm), 1.0))

    @staticmethod
    def _nutrition_priority_multiplier(nutrition_priority: str, food: FoodCandidate) -> float:
        """Boost score based on nutrition priority preference."""
        protein_norm = min(food.nutrition.protein / 50, 1.0)
        carbs_norm = min(food.nutrition.carbs / 80, 1.0)
        fat_norm = min(food.nutrition.fat / 35, 1.0)
        fiber_norm = min(food.nutrition.fiber / 14, 1.0)

        priority_multipliers = {
            "BALANCED": 1.0,
            "HIGH_PROTEIN": 1.0 + (0.5 * protein_norm),
            "HIGH_CARBS": 1.0 + (0.5 * carbs_norm),
            "HIGH_FAT": 1.0 + (0.5 * fat_norm),
            "HIGH_FIBER": 1.0 + (0.5 * fiber_norm),
        }
        return priority_multipliers.get(nutrition_priority, 1.0)

    @staticmethod
    def _goal_multiplier(goal_type: str, food: FoodCandidate) -> float:
        calories_norm = min(food.nutrition.calories / 700, 1.0)
        protein_norm = min(food.nutrition.protein / 50, 1.0)
        fat_norm = min(food.nutrition.fat / 35, 1.0)
        fiber_norm = min(food.nutrition.fiber / 14, 1.0)

        adjustments = {
            "WEIGHT_LOSS": (-0.3 * calories_norm) + (0.1 * protein_norm) + (0.2 * fiber_norm),
            "WEIGHT_GAIN": (0.1 * calories_norm) + (0.4 * protein_norm),
            "MAINTENANCE": 0.0,
        }
        return max(0.25, 1 + adjustments.get(goal_type, 0.0) - (0.05 * fat_norm if goal_type == "WEIGHT_LOSS" else 0.0))

    @staticmethod
    def _dynamic_weights(
        total_logs: int, goal_type: str, has_cf: bool, nutrition_priority: str = "BALANCED"
    ) -> tuple[float, float, float]:
        # Base weights by user maturity and collaborative filtering availability
        if not has_cf:
            if total_logs < 10:
                alpha, beta, gamma = 0.95, 0.0, 0.05
            elif total_logs <= 60:
                alpha, beta, gamma = 0.90, 0.0, 0.10
            else:
                alpha, beta, gamma = 0.85, 0.0, 0.15
        else:
            if total_logs < 10:
                alpha, beta, gamma = 0.90, 0.05, 0.05
            elif total_logs <= 60:
                alpha, beta, gamma = 0.70, 0.20, 0.10
            else:
                alpha, beta, gamma = 0.55, 0.35, 0.10

        # BOOST khi có nutrition_priority
        if nutrition_priority != "BALANCED":
            boost = 0.30 if nutrition_priority in ("HIGH_PROTEIN", "HIGH_FIBER") else 0.25
            alpha = min(alpha + boost, 0.95)
            beta = max(beta - boost * 0.85, 0.03)

        if goal_type == "STRICT_DIET":
            alpha = min(alpha + 0.10, 1.0)
            beta = max(beta - 0.10, 0.0)

        return alpha, beta, gamma