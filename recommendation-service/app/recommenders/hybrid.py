from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from app.db.repositories.recommendation_repository import FoodCandidate, UserContextRecord


@dataclass
class ScoreBreakdown:
    content_score: float
    collaborative_score: float
    popular_score: float
    profile_score: float
    repeat_penalty: float
    final_score: float
    meal_affinity: float
    alpha: float
    beta: float
    gamma: float
    delta: float  # weight for popular
    epsilon: float  # weight for profile


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

        # Tích hợp Popular + UserProfile scores
        popular_score = self._popular_score(food)
        profile_score = self._profile_score(food, context)

        alpha, beta, gamma, delta, epsilon = self._dynamic_weights(
            context.total_logs, context.goal_type, collaborative_score > 0, nutrition_priority
        )

        # Hybrid score với popular và profile
        hybrid_score = (
            alpha * content_score
            + beta * collaborative_score
            + delta * popular_score
            + epsilon * profile_score
            - gamma * repeat_penalty
        )
        final_score = max(0.0, min(hybrid_score, 1.0))

        return ScoreBreakdown(
            content_score=content_score,
            collaborative_score=collaborative_score,
            popular_score=popular_score,
            profile_score=profile_score,
            repeat_penalty=repeat_penalty,
            final_score=final_score,
            meal_affinity=food.meal_affinity.get(meal_type, 0.25),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            epsilon=epsilon,
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

        # Nutrition priority boost (additive approach - mượt hơn)
        priority_boost = 0.0
        if nutrition_priority != "BALANCED":
            multiplier = self._nutrition_priority_multiplier(nutrition_priority, food)
            # Convert multiplicative boost to additive: (multiplier - 1) * 0.4
            priority_boost = (multiplier - 1.0) * 0.4

        final_content_score = base_score * goal_multiplier * meal_affinity + priority_boost
        return max(0.0, min(final_content_score, 1.0))

    @staticmethod
    def _repeat_penalty(food_id: int, context: UserContextRecord, threshold: int = 3) -> float:
        """Exponential repeat penalty - phạt mạnh hơn khi count cao."""
        count = context.repeat_counts.get(food_id, 0)
        # Exponential decay: 1 - (0.85 ** count)
        # count=0 -> 0, count=1 -> 0.15, count=2 -> 0.28, count=3 -> 0.39, etc.
        penalty = 1.0 - (0.85 ** count)
        return max(0.0, min(penalty, 1.0))

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
    def _popular_score(food: FoodCandidate) -> float:
        """Tính điểm popularity dựa trên số lần món ăn được chọn."""
        # Normalize: 0-20 count -> 0-1 score
        return min(food.popularity_count / 20, 1.0)

    @staticmethod
    def _profile_score(food: FoodCandidate, context: UserContextRecord) -> float:
        """Tính điểm user profile preference dựa trên category."""
        category_name = (food.category.name or "").lower()
        if not category_name:
            return 0.0
        category_frequency = context.category_scores.get(category_name, 0)
        # Normalize: 0-5 count -> 0-1 score
        return min(category_frequency / 5, 1.0)

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
            # STRICT_DIET: phạt mạnh calories và fat cao, thưởng protein và fiber
            "STRICT_DIET": (-0.5 * calories_norm) + (0.15 * protein_norm) + (0.25 * fiber_norm) - (0.10 * fat_norm),
        }
        return max(0.25, 1 + adjustments.get(goal_type, 0.0) - (0.05 * fat_norm if goal_type in ("WEIGHT_LOSS", "STRICT_DIET") else 0.0))

    @staticmethod
    def _dynamic_weights(
        total_logs: int, goal_type: str, has_cf: bool, nutrition_priority: str = "BALANCED"
    ) -> tuple[float, float, float, float, float]:
        # Base weights by user maturity and collaborative filtering availability
        # alpha = content, beta = collaborative, gamma = repeat penalty
        # delta = popular, epsilon = profile
        if not has_cf:
            if total_logs < 10:
                alpha, beta, gamma = 0.90, 0.0, 0.05
            elif total_logs <= 60:
                alpha, beta, gamma = 0.85, 0.0, 0.10
            else:
                alpha, beta, gamma = 0.80, 0.0, 0.15
        else:
            if total_logs < 10:
                alpha, beta, gamma = 0.85, 0.05, 0.05
            elif total_logs <= 60:
                alpha, beta, gamma = 0.65, 0.20, 0.10
            else:
                alpha, beta, gamma = 0.50, 0.35, 0.10

        # Popular và Profile weights theo recommend của user
        delta = 0.06  # popular weight (tăng nhẹ để thấy hiệu quả)
        epsilon = 0.07  # profile weight (sở thích cá nhân quan trọng hơn)

        # BOOST khi có nutrition_priority
        if nutrition_priority != "BALANCED":
            boost = 0.30 if nutrition_priority in ("HIGH_PROTEIN", "HIGH_FIBER") else 0.25
            alpha = min(alpha + boost, 0.90)
            beta = max(beta - boost * 0.85, 0.03)
            delta = max(delta - 0.02, 0.04)  # Giảm ít hơn vì base đã cao hơn
            epsilon = max(epsilon - 0.02, 0.05)

        if goal_type == "STRICT_DIET":
            alpha = min(alpha + 0.10, 0.95)
            beta = max(beta - 0.10, 0.0)
            delta = max(delta - 0.02, 0.03)
            epsilon = max(epsilon - 0.02, 0.03)

        # Normalize positive weights so they sum to 1.0 (gamma is subtracted separately)
        total_positive = alpha + beta + delta + epsilon
        if total_positive > 0:
            alpha = alpha / total_positive
            beta = beta / total_positive
            delta = delta / total_positive
            epsilon = epsilon / total_positive

        return alpha, beta, gamma, delta, epsilon