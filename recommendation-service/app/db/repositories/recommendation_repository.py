from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


MEAL_TYPES = ("BREAKFAST", "LUNCH", "DINNER", "SNACK")


@dataclass
class NutritionVector:
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0

    def as_list(self) -> list[float]:
        return [self.calories, self.protein, self.carbs, self.fat, self.fiber]


@dataclass
class CategoryRecord:
    category_id: int | None = None
    name: str | None = None


@dataclass
class FoodCandidate:
    food_id: int
    food_name: str
    description: str | None = None
    image_url: str | None = None
    created_at: datetime | None = None
    category: CategoryRecord = field(default_factory=CategoryRecord)
    nutrition: NutritionVector = field(default_factory=NutritionVector)
    allergens: set[str] = field(default_factory=set)
    meal_affinity: dict[str, float] = field(default_factory=dict)
    popularity_count: int = 0


@dataclass
class UserContextRecord:
    user_id: int | None
    goal_type: str = "MAINTENANCE"
    target_nutrition: NutritionVector = field(default_factory=NutritionVector)
    consumed_today: NutritionVector = field(default_factory=NutritionVector)
    remaining_nutrition: NutritionVector = field(default_factory=NutritionVector)
    burned_calories_today: float = 0.0
    allergy_warnings: list[str] = field(default_factory=list)
    repeat_counts: dict[int, int] = field(default_factory=dict)
    total_logs: int = 0


class RecommendationRepository:
    def __init__(self, engine: Engine, schema: str) -> None:
        self.engine = engine
        self.schema = schema

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False

    def load_user_context(self, user_id: int | None, current_time: datetime) -> UserContextRecord:
        context = UserContextRecord(user_id=user_id)
        if user_id is None:
            return context

        current_date = current_time.date()
        goal_row = self._fetch_active_goal(user_id, current_date)
        consumed_row = self._fetch_consumed_today(user_id, current_date)
        allergy_names = self._fetch_user_allergies(user_id)
        repeat_counts = self._fetch_repeat_counts(user_id, current_date)
        total_logs = self._fetch_total_logs(user_id)

        target = NutritionVector(
            calories=float((goal_row or {}).get("targetCalories") or 0.0),
            protein=float((goal_row or {}).get("targetProtein") or 0.0),
            carbs=float((goal_row or {}).get("targetCarbs") or 0.0),
            fat=float((goal_row or {}).get("targetFat") or 0.0),
            fiber=float((goal_row or {}).get("targetFiber") or 0.0),
        )
        consumed = NutritionVector(
            calories=float((consumed_row or {}).get("calories") or 0.0),
            protein=float((consumed_row or {}).get("protein") or 0.0),
            carbs=float((consumed_row or {}).get("carbs") or 0.0),
            fat=float((consumed_row or {}).get("fat") or 0.0),
            fiber=float((consumed_row or {}).get("fiber") or 0.0),
        )
        burned_calories = 0.0
        remaining = NutritionVector(
            calories=max(target.calories + burned_calories - consumed.calories, 0.0),
            protein=max(target.protein - consumed.protein, 0.0),
            carbs=max(target.carbs - consumed.carbs, 0.0),
            fat=max(target.fat - consumed.fat, 0.0),
            fiber=max(target.fiber - consumed.fiber, 0.0),
        )

        context.goal_type = str((goal_row or {}).get("goalType") or "MAINTENANCE")
        context.target_nutrition = target
        context.consumed_today = consumed
        context.remaining_nutrition = remaining
        context.burned_calories_today = burned_calories
        context.allergy_warnings = allergy_names
        context.repeat_counts = repeat_counts
        context.total_logs = total_logs
        return context

    def load_food_candidates(self, limit: int) -> list[FoodCandidate]:
        sql = text(
            f"""
            SELECT
                f.id AS food_id,
                f."foodName" AS food_name,
                f.description,
                f."imageUrl" AS image_url,
                f."createdAt" AS created_at,
                fc.id AS category_id,
                fc.name AS category_name,
                COALESCE(MAX(CASE WHEN LOWER(n.name) IN ('calories', 'energy') THEN fnv.value END), 0) AS calories,
                COALESCE(MAX(CASE WHEN LOWER(n.name) = 'protein' THEN fnv.value END), 0) AS protein,
                COALESCE(MAX(CASE WHEN LOWER(n.name) IN ('carbs', 'carbohydrates') THEN fnv.value END), 0) AS carbs,
                COALESCE(MAX(CASE WHEN LOWER(n.name) IN ('fat', 'total fat') THEN fnv.value END), 0) AS fat,
                COALESCE(MAX(CASE WHEN LOWER(n.name) IN ('fiber', 'fibre', 'dietary fiber', 'dietary fibre') THEN fnv.value END), 0) AS fiber,
                COALESCE(ARRAY_AGG(DISTINCT a.name) FILTER (WHERE a.name IS NOT NULL), ARRAY[]::text[]) AS allergens
            FROM {self._table('foods')} f
            LEFT JOIN {self._table('food_categories')} fc ON fc.id = f."categoryId"
            LEFT JOIN {self._table('food_nutrition_profiles')} fnp ON fnp."foodId" = f.id
            LEFT JOIN {self._table('food_nutrition_values')} fnv ON fnv."foodNutritionProfileId" = fnp.id
            LEFT JOIN {self._table('nutrients')} n ON n.id = fnv."nutrientId"
            LEFT JOIN {self._table('food_ingredients')} fi ON fi."foodId" = f.id
            LEFT JOIN {self._table('ingredient_allergens')} ia ON ia."ingredientId" = fi."ingredientId"
            LEFT JOIN {self._table('allergens')} a ON a.id = ia."allergenId"
            GROUP BY f.id, f."foodName", f.description, f."imageUrl", f."createdAt", fc.id, fc.name
            ORDER BY f."createdAt" DESC, f.id DESC
            LIMIT :limit
            """
        )
        meal_stats = self._fetch_food_meal_stats()
        with self.engine.connect() as connection:
            rows = connection.execute(sql, {"limit": limit}).mappings().all()

        candidates: list[FoodCandidate] = []
        for row in rows:
            food_id = int(row["food_id"])
            stats = meal_stats.get(food_id, {})
            candidates.append(
                FoodCandidate(
                    food_id=food_id,
                    food_name=str(row["food_name"]),
                    description=row.get("description"),
                    image_url=row.get("image_url"),
                    created_at=row.get("created_at"),
                    category=CategoryRecord(
                        category_id=row.get("category_id"),
                        name=row.get("category_name"),
                    ),
                    nutrition=NutritionVector(
                        calories=float(row.get("calories") or 0.0),
                        protein=float(row.get("protein") or 0.0),
                        carbs=float(row.get("carbs") or 0.0),
                        fat=float(row.get("fat") or 0.0),
                        fiber=float(row.get("fiber") or 0.0),
                    ),
                    allergens={str(item) for item in (row.get("allergens") or []) if item},
                    meal_affinity=stats.get("affinity", self._default_affinity()),
                    popularity_count=int(stats.get("total_count", 0)),
                )
            )
        return candidates

    def load_user_item_frequencies(self, user_id: int | None, limit: int = 200) -> dict[int, float]:
        if user_id is None:
            return {}
        sql = text(
            f"""
            SELECT mi."foodId" AS food_id, COUNT(*) AS frequency
            FROM {self._table('meal_items')} mi
            JOIN {self._table('meals')} m ON m.id = mi."mealId"
            JOIN {self._table('daily_logs')} dl ON dl.id = m."dailyLogId"
            WHERE dl."userId" = :user_id
            GROUP BY mi."foodId"
            ORDER BY frequency DESC, mi."foodId" ASC
            LIMIT :limit
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(sql, {"user_id": user_id, "limit": limit}).mappings().all()
            return {int(row["food_id"]): float(row["frequency"]) for row in rows}
        except Exception:
            return {}

    def load_candidate_collaborative_scores(
        self,
        user_id: int | None,
        candidate_ids: list[int],
        neighbor_limit: int = 20,
        min_shared_items: int = 1,
        history_limit: int = 200,
    ) -> dict[int, float]:
        if user_id is None or not candidate_ids:
            return {}

        sql = text(
            f"""
            WITH user_history AS (
                SELECT mi."foodId" AS food_id, COUNT(*)::float AS freq
                FROM {self._table('meal_items')} mi
                JOIN {self._table('meals')} m ON m.id = mi."mealId"
                JOIN {self._table('daily_logs')} dl ON dl.id = m."dailyLogId"
                WHERE dl."userId" = :user_id
                GROUP BY mi."foodId"
                ORDER BY COUNT(*) DESC
                LIMIT :history_limit
            ),
            neighbors AS (
                SELECT dl."userId" AS neighbor_id,
                       COUNT(DISTINCT mi."foodId") AS shared_items,
                       SUM(LOG(1 + uh.freq))::float AS similarity
                FROM user_history uh
                JOIN {self._table('meal_items')} mi ON mi."foodId" = uh.food_id
                JOIN {self._table('meals')} m ON m.id = mi."mealId"
                JOIN {self._table('daily_logs')} dl ON dl.id = m."dailyLogId"
                WHERE dl."userId" <> :user_id
                GROUP BY dl."userId"
                HAVING COUNT(DISTINCT mi."foodId") >= :min_shared_items
                ORDER BY similarity DESC, shared_items DESC
                LIMIT :neighbor_limit
            ),
            neighbor_foods AS (
                SELECT mi."foodId" AS food_id,
                       SUM(LOG(1 + mi.quantity) * n.similarity)::float AS weighted_score
                FROM neighbors n
                JOIN {self._table('daily_logs')} dl ON dl."userId" = n.neighbor_id
                JOIN {self._table('meals')} m ON m."dailyLogId" = dl.id
                JOIN {self._table('meal_items')} mi ON mi."mealId" = m.id
                WHERE mi."foodId" = ANY(:candidate_ids)
                GROUP BY mi."foodId"
            )
            SELECT food_id, weighted_score
            FROM neighbor_foods
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(
                    sql,
                    {
                        "user_id": user_id,
                        "candidate_ids": candidate_ids,
                        "neighbor_limit": neighbor_limit,
                        "min_shared_items": min_shared_items,
                        "history_limit": history_limit,
                    },
                ).mappings().all()
            raw = {int(row["food_id"]): float(row["weighted_score"]) for row in rows}
            return self._normalize_scores(raw)
        except Exception:
            return {}

    def summarize_food_catalog(self, limit: int = 500) -> dict[str, Any]:
        foods = self.load_food_candidates(limit)
        return {
            "foods": len(foods),
            "categories": len({food.category.name for food in foods if food.category.name}),
            "foods_with_nutrition": sum(1 for food in foods if food.nutrition.calories > 0),
            "new_items_7d": sum(1 for food in foods if food.created_at and self._normalize_dt(food.created_at) >= datetime.now().replace(microsecond=0) - timedelta(days=7)),
        }

    def _fetch_active_goal(self, user_id: int, current_date: date) -> dict[str, Any] | None:
        target_fiber_expr = 'ng."targetFiber"' if self._has_column("nutrition_goals", "targetFiber") else '0'
        sql = text(
            f"""
            SELECT
                ng."goalType",
                ng."targetCalories",
                ng."targetProtein",
                ng."targetCarbs",
                ng."targetFat",
                {target_fiber_expr} AS "targetFiber"
            FROM {self._table('nutrition_goals')} ng
            WHERE ng."userId" = :user_id
              AND DATE(ng."startDay") <= :current_date
              AND DATE(ng."endDate") >= :current_date
            ORDER BY ng."startDay" DESC, ng.id DESC
            LIMIT 1
            """
        )
        try:
            with self.engine.connect() as connection:
                row = connection.execute(sql, {"user_id": user_id, "current_date": current_date}).mappings().first()
            return dict(row) if row else None
        except Exception:
            return None

    def _fetch_consumed_today(self, user_id: int, current_date: date) -> dict[str, Any] | None:
        sql = text(
            f"""
            SELECT
                COALESCE(SUM(mi.calories), 0) AS calories,
                COALESCE(SUM(mi.protein), 0) AS protein,
                COALESCE(SUM(mi.carbs), 0) AS carbs,
                COALESCE(SUM(mi.fat), 0) AS fat,
                COALESCE(SUM(mi.fiber), 0) AS fiber
            FROM {self._table('daily_logs')} dl
            JOIN {self._table('meals')} m ON m."dailyLogId" = dl.id
            JOIN {self._table('meal_items')} mi ON mi."mealId" = m.id
            WHERE dl."userId" = :user_id
              AND dl."logDate" = :current_date
            """
        )
        try:
            with self.engine.connect() as connection:
                row = connection.execute(sql, {"user_id": user_id, "current_date": current_date}).mappings().first()
            return dict(row) if row else None
        except Exception:
            return None

    def _fetch_user_allergies(self, user_id: int) -> list[str]:
        sql = text(
            f"""
            SELECT a.name
            FROM {self._table('user_allergies')} ua
            JOIN {self._table('allergens')} a ON a.id = ua."allergenId"
            WHERE ua."userId" = :user_id
            ORDER BY a.name ASC
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(sql, {"user_id": user_id}).mappings().all()
            return [str(row["name"]) for row in rows if row.get("name")]
        except Exception:
            return []

    def _fetch_repeat_counts(self, user_id: int, current_date: date) -> dict[int, int]:
        start_date = current_date - timedelta(days=6)
        sql = text(
            f"""
            SELECT mi."foodId" AS food_id, COUNT(*) AS total_count
            FROM {self._table('daily_logs')} dl
            JOIN {self._table('meals')} m ON m."dailyLogId" = dl.id
            JOIN {self._table('meal_items')} mi ON mi."mealId" = m.id
            WHERE dl."userId" = :user_id
              AND dl."logDate" BETWEEN :start_date AND :current_date
            GROUP BY mi."foodId"
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(
                    sql,
                    {"user_id": user_id, "start_date": start_date, "current_date": current_date},
                ).mappings().all()
            return {int(row["food_id"]): int(row["total_count"]) for row in rows}
        except Exception:
            return {}

    def _fetch_total_logs(self, user_id: int) -> int:
        sql = text(
            f"""
            SELECT COUNT(*) AS total_count
            FROM {self._table('meal_items')} mi
            JOIN {self._table('meals')} m ON m.id = mi."mealId"
            JOIN {self._table('daily_logs')} dl ON dl.id = m."dailyLogId"
            WHERE dl."userId" = :user_id
            """
        )
        try:
            with self.engine.connect() as connection:
                row = connection.execute(sql, {"user_id": user_id}).mappings().first()
            return int((row or {}).get("total_count") or 0)
        except Exception:
            return 0

    def _fetch_food_meal_stats(self) -> dict[int, dict[str, Any]]:
        sql = text(
            f"""
            SELECT mi."foodId" AS food_id, m."mealType" AS meal_type, COUNT(*) AS total_count
            FROM {self._table('meal_items')} mi
            JOIN {self._table('meals')} m ON m.id = mi."mealId"
            GROUP BY mi."foodId", m."mealType"
            """
        )
        try:
            with self.engine.connect() as connection:
                rows = connection.execute(sql).mappings().all()
        except Exception:
            return {}

        grouped: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[int, int] = defaultdict(int)
        for row in rows:
            food_id = int(row["food_id"])
            meal_type = str(row["meal_type"])
            count = int(row["total_count"])
            grouped[food_id][meal_type] += count
            totals[food_id] += count

        results: dict[int, dict[str, Any]] = {}
        for food_id, counts in grouped.items():
            total = totals[food_id]
            affinity = {
                meal_type: (counts.get(meal_type, 0) + 1) / (total + len(MEAL_TYPES))
                for meal_type in MEAL_TYPES
            }
            results[food_id] = {"affinity": affinity, "total_count": total}
        return results

    @staticmethod
    def _default_affinity() -> dict[str, float]:
        base = 1 / len(MEAL_TYPES)
        return {meal_type: base for meal_type in MEAL_TYPES}

    @staticmethod
    def _normalize_scores(raw: dict[int, float]) -> dict[int, float]:
        if not raw:
            return {}
        minimum = min(raw.values())
        maximum = max(raw.values())
        if minimum == maximum:
            return {key: 1.0 for key in raw}
        return {key: (value - minimum) / (maximum - minimum) for key, value in raw.items()}

    @staticmethod
    def _normalize_dt(value: datetime) -> datetime:
        return value.replace(tzinfo=None) if value.tzinfo else value

    @lru_cache(maxsize=32)
    def _columns_for_table(self, table_name: str) -> set[str]:
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name, schema=self.schema)
            return {str(column["name"]) for column in columns}
        except Exception:
            return set()

    def _has_column(self, table_name: str, column_name: str) -> bool:
        return column_name in self._columns_for_table(table_name)

    def _table(self, table_name: str) -> str:
        return f'"{self.schema}"."{table_name}"'
