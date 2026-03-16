from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    server_host: str = Field(..., alias="SERVER_HOST")
    server_port: int = Field(..., alias="SERVER_PORT")
    database_url: str = Field(..., alias="DATABASE_URL")
    api_prefix: str = "/v1"
    app_name: str = "Food Recommendation Service"
    db_schema: str = "public"
    recommendation_cache_ttl_seconds: int = 120
    candidate_limit: int = 200
    default_limit: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    @property
    def normalized_database_url(self) -> str:
        if "?" not in self.database_url:
            return self.database_url
        base, _, query = self.database_url.partition("?")
        parts = [part for part in query.split("&") if not part.startswith("schema=")]
        return base if not parts else f"{base}?{'&'.join(parts)}"

    @property
    def resolved_db_schema(self) -> str:
        if "schema=" in self.database_url:
            for part in self.database_url.split("?", maxsplit=1)[1].split("&"):
                if part.startswith("schema="):
                    return part.split("=", maxsplit=1)[1]
        return self.db_schema


@lru_cache
def get_settings() -> Settings:
    return Settings()