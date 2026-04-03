from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.db.repositories.recommendation_repository import RecommendationRepository


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.normalized_database_url, pool_pre_ping=True)


@lru_cache
def get_repository() -> RecommendationRepository:
    settings = get_settings()
    return RecommendationRepository(get_engine(), settings.resolved_db_schema)