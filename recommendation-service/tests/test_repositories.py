from unittest.mock import MagicMock

from app.db.repositories.recommendation_repository import RecommendationRepository


def test_ping_returns_false_when_connection_fails():
    engine = MagicMock()
    engine.connect.side_effect = Exception("db down")
    repository = RecommendationRepository(engine=engine, schema="public")

    assert repository.ping() is False