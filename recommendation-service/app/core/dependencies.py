from functools import lru_cache

from app.core.config import get_settings
from app.db.session import get_repository
from app.services.cache_service import CacheService
from app.services.recommendation_service import RecommendationService


@lru_cache
def get_cache_service() -> CacheService:
    settings = get_settings()
    return CacheService(ttl_seconds=settings.recommendation_cache_ttl_seconds)


@lru_cache
def get_recommendation_service() -> RecommendationService:
    settings = get_settings()
    return RecommendationService(
        repository=get_repository(),
        settings=settings,
        cache_service=get_cache_service(),
    )