from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import MetaData, Table, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


FOOD_TABLE_CANDIDATES = [
    "foods",
    "food",
    "menu_items",
    "menu_item",
    "dishes",
    "dish",
    "items",
    "products",
]

INTERACTION_TABLE_CANDIDATES = [
    "user_food_events",
    "food_interactions",
    "favorites",
    "favourites",
    "ratings",
    "reviews",
    "cart_items",
    "order_items",
]

FIELD_CANDIDATES = {
    "food_id": ["id", "food_id", "dish_id", "item_id", "product_id"],
    "name": ["name", "food_name", "dish_name", "item_name", "title"],
    "category": ["category", "category_name", "food_category", "type"],
    "cuisine": ["cuisine", "cuisine_type", "style"],
    "price": ["price", "unit_price", "sale_price", "base_price"],
    "description": ["description", "summary", "details"],
    "tags": ["tags", "dietary_tags", "labels"],
    "available": ["is_available", "available", "in_stock", "status"],
    "popularity": ["popularity_score", "order_count", "favorite_count", "views", "rating_count"],
    "rating": ["rating", "avg_rating", "score"],
    "meal_time": ["meal_time", "served_at", "day_part"],
}

INTERACTION_FIELD_CANDIDATES = {
    "user_id": ["user_id", "customer_id", "account_id"],
    "food_id": ["food_id", "dish_id", "item_id", "product_id"],
    "event_type": ["event_type", "action", "interaction_type", "status"],
    "rating": ["rating", "score"],
}


@dataclass
class FoodRecord:
    food_id: str
    name: str
    category: str | None = None
    cuisine: str | None = None
    price: float | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    meal_time: str | None = None
    available: bool = True
    popularity: float = 0.0
    rating: float = 0.0


@dataclass
class InteractionRecord:
    food_id: str
    event_type: str
    weight: float


class RecommendationRepository:
    def __init__(self, engine: Engine, schema: str) -> None:
        self.engine = engine
        self.schema = schema
        self.metadata = MetaData(schema=schema)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            return True
        except Exception:
            return False

    def load_catalog(self, limit: int) -> list[FoodRecord]:
        table_name = self._find_first_existing_table(FOOD_TABLE_CANDIDATES)
        if not table_name:
            return []

        table = Table(table_name, self.metadata, autoload_with=self.engine)
        column_map = self._build_column_map(list(table.columns.keys()), FIELD_CANDIDATES)
        selected_columns = [
            table.c[column_name].label(target_name)
            for target_name, column_name in column_map.items()
            if column_name in table.c
        ]
        if not selected_columns:
            return []

        stmt = select(*selected_columns).limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(stmt).mappings().all()
        return [self._map_food_record(row) for row in rows]

    def load_user_history(self, user_id: str | None) -> list[InteractionRecord]:
        if not user_id:
            return []

        table_name = self._find_first_existing_table(INTERACTION_TABLE_CANDIDATES)
        if not table_name:
            return []

        table = Table(table_name, self.metadata, autoload_with=self.engine)
        column_map = self._build_column_map(list(table.columns.keys()), INTERACTION_FIELD_CANDIDATES)
        if "user_id" not in column_map or "food_id" not in column_map:
            return []

        selected_columns = [table.c[column_map["food_id"]].label("food_id")]
        if "event_type" in column_map:
            selected_columns.append(table.c[column_map["event_type"]].label("event_type"))
        if "rating" in column_map:
            selected_columns.append(table.c[column_map["rating"]].label("rating"))

        stmt = (
            select(*selected_columns)
            .where(table.c[column_map["user_id"]] == user_id)
            .limit(100)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(stmt).mappings().all()
        return [self._map_interaction_record(row) for row in rows]

    def summarize_catalog(self, catalog: list[FoodRecord]) -> dict[str, Counter]:
        categories = Counter(item.category for item in catalog if item.category)
        cuisines = Counter(item.cuisine for item in catalog if item.cuisine)
        meal_times = Counter(item.meal_time for item in catalog if item.meal_time)
        return {
            "categories": categories,
            "cuisines": cuisines,
            "meal_times": meal_times,
        }

    def _find_first_existing_table(self, candidates: list[str]) -> str | None:
        try:
            inspector = inspect(self.engine)
            tables = set(inspector.get_table_names(schema=self.schema))
        except Exception:
            return None
        for candidate in candidates:
            if candidate in tables:
                return candidate
        return None

    @staticmethod
    def _build_column_map(columns: list[str], field_candidates: dict[str, list[str]]) -> dict[str, str]:
        column_set = set(columns)
        mapping: dict[str, str] = {}
        for target, candidates in field_candidates.items():
            for candidate in candidates:
                if candidate in column_set:
                    mapping[target] = candidate
                    break
        return mapping

    @staticmethod
    def _normalize_tags(raw_value: Any) -> list[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, list):
            return [str(item).strip().lower() for item in raw_value if str(item).strip()]
        return [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]

    def _map_food_record(self, row: Any) -> FoodRecord:
        available_value = row.get("available", True)
        if isinstance(available_value, str):
            available = available_value.lower() not in {"false", "0", "inactive", "out_of_stock"}
        else:
            available = bool(available_value)
        return FoodRecord(
            food_id=str(row.get("food_id")),
            name=str(row.get("name") or row.get("food_id")),
            category=row.get("category"),
            cuisine=row.get("cuisine"),
            price=float(row["price"]) if row.get("price") is not None else None,
            description=row.get("description"),
            tags=self._normalize_tags(row.get("tags")),
            meal_time=row.get("meal_time"),
            available=available,
            popularity=float(row["popularity"]) if row.get("popularity") is not None else 0.0,
            rating=float(row["rating"]) if row.get("rating") is not None else 0.0,
        )

    @staticmethod
    def _map_interaction_record(row: Any) -> InteractionRecord:
        event_type = str(row.get("event_type") or "interaction")
        rating = float(row["rating"]) if row.get("rating") is not None else None
        weight_by_event = {
            "impression": 0.1,
            "click": 0.4,
            "favorite": 0.8,
            "order": 1.0,
            "rating": rating / 5 if rating is not None else 0.5,
            "interaction": 0.3,
        }
        return InteractionRecord(
            food_id=str(row["food_id"]),
            event_type=event_type,
            weight=weight_by_event.get(event_type, 0.3),
        )