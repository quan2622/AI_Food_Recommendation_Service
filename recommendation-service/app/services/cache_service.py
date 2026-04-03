from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


class CacheService:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, CacheEntry[object]] = {}

    def get(self, key: str) -> object | None:
        entry = self._store.get(key)
        if not entry:
            return None
        if entry.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: object) -> None:
        self._store[key] = CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)