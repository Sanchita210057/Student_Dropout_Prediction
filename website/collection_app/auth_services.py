from __future__ import annotations

import hashlib

from django.core.cache import cache


def _key(identifier: str, address: str) -> str:
    return "login-attempt:" + hashlib.sha256(f"{identifier.casefold()}|{address}".encode("utf-8")).hexdigest()


def allow_login_attempt(identifier: str, address: str, *, limit: int = 8, window: int = 900) -> tuple[bool, str]:
    key = _key(identifier, address)
    return cache.get(key, 0) < limit, key


def record_login_failure(key: str, *, window: int = 900) -> None:
    if not cache.add(key, 1, timeout=window):
        try:
            cache.incr(key)
        except ValueError:
            cache.set(key, 1, timeout=window)


def clear_login_failures(key: str) -> None:
    cache.delete(key)
