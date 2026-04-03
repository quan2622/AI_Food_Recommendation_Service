from datetime import date

from unittest.mock import MagicMock

from app.db.repositories.recommendation_repository import RecommendationRepository


def test_ping_returns_false_when_connection_fails():
    engine = MagicMock()
    engine.connect.side_effect = Exception("db down")
    repository = RecommendationRepository(engine=engine, schema="public")

    assert repository.ping() is False


def test_fetch_consumed_today_matches_meal_datetime_or_log_date():
    row = {"calories": 1152.5, "protein": 114.44, "carbs": 143.07, "fat": 28.67, "fiber": 20.33}

    connection = MagicMock()
    connection.execute.return_value.mappings.return_value.first.return_value = row
    context_manager = MagicMock()
    context_manager.__enter__.return_value = connection
    context_manager.__exit__.return_value = False

    engine = MagicMock()
    engine.connect.return_value = context_manager

    repository = RecommendationRepository(engine=engine, schema="public")

    result = repository._fetch_consumed_today(user_id=1, current_date=date(2026, 3, 19))

    assert result == row

    statement = connection.execute.call_args.args[0]
    params = connection.execute.call_args.args[1]
    sql_text = str(statement)

    assert 'DATE(m."mealDateTime") = :current_date' in sql_text
    assert 'DATE(dl."logDate") = :current_date' in sql_text
    assert params == {"user_id": 1, "current_date": date(2026, 3, 19)}
