from datetime import datetime


def resolve_meal_time(explicit_meal_time: str | None, now: datetime | None = None) -> str:
    if explicit_meal_time:
        return explicit_meal_time
    current_time = now or datetime.utcnow()
    hour = current_time.hour
    if 5 <= hour < 11:
        return "breakfast"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 21:
        return "dinner"
    return "snack"