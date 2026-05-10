"""DataGolf API client.

DataGolf provides high-quality golf predictions. We use them as a
Tier-1 truth source for the golf model, falling back to The Odds API
when a player isn't found.

Free tier requires registration at datagolf.com. Set the API key via
the DATAGOLF_API_KEY environment variable (Render: Environment tab).
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import structlog

log = structlog.get_logger(__name__)

API_BASE = "https://feeds.datagolf.com"
IN_PLAY_TTL_SEC = 300       # 5 min
PRE_TOURNEY_TTL_SEC = 1800  # 30 min

# module-level caches keyed by tour
_in_play_cache: dict[str, tuple[float, list[dict]]] = {}
_pre_cache: dict[str, tuple[float, list[dict]]] = {}
_locks: dict[str, asyncio.Lock] = {}

# Single warning-on-startup pattern: log only the first time the key is missing
_warned_no_key = False


def _api_key() -> str | None:
    return os.environ.get("DATAGOLF_API_KEY") or None


def _warn_no_key_once() -> None:
    global _warned_no_key
    if not _warned_no_key:
        log.warning(
            "datagolf.no_api_key",
            msg="DATAGOLF_API_KEY env var not set; DataGolf disabled",
        )
        _warned_no_key = True


def _extract_players(data: object) -> list[dict]:
    """The endpoint returns {"data": [...]} or sometimes just a list directly.
    Handle both: if response is a dict, look for 'data' or 'players' key;
    else use as-is.
    """
    if isinstance(data, dict):
        for key in ("data", "players"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        return []
    if isinstance(data, list):
        return data
    return []


async def _fetch_cached(
    *,
    tour: str,
    endpoint: str,
    cache: dict[str, tuple[float, list[dict]]],
    ttl: int,
    lock_key: str,
) -> list[dict] | None:
    api_key = _api_key()
    if not api_key:
        _warn_no_key_once()
        return None

    cache_key = tour
    cached = cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < ttl:
        return cached[1]

    lock = _locks.setdefault(lock_key, asyncio.Lock())
    async with lock:
        cached = cache.get(cache_key)
        if cached and (time.time() - cached[0]) < ttl:
            return cached[1]
        url = f"{API_BASE}/preds/{endpoint}"
        params = {
            "tour": tour,
            "dead_heat": "no",
            "odds_format": "percent",
            "file_format": "json",
            "key": api_key,
        }
        # pre-tournament expects an extra add_position param (can be empty)
        if endpoint == "pre-tournament":
            params["add_position"] = ""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning(
                "datagolf.fetch_failed",
                err=str(e)[:120],
                endpoint=endpoint,
                tour=tour,
            )
            return None
        players = _extract_players(data)
        log.info(
            "datagolf.fetched",
            endpoint=endpoint,
            tour=tour,
            players=len(players),
        )
        cache[cache_key] = (time.time(), players)
        return players


async def fetch_in_play(tour: str = "pga") -> list[dict] | None:
    """Returns list of player prediction dicts, or None on missing key /
    network error. Each player dict has at least 'player_name' and 'win'
    fields. Cached for 5 min per tour.
    """
    return await _fetch_cached(
        tour=tour,
        endpoint="in-play",
        cache=_in_play_cache,
        ttl=IN_PLAY_TTL_SEC,
        lock_key=f"in_play:{tour}",
    )


async def fetch_pre_tournament(tour: str = "pga") -> list[dict] | None:
    """Pre-tournament predictions. Cached for 30 min."""
    return await _fetch_cached(
        tour=tour,
        endpoint="pre-tournament",
        cache=_pre_cache,
        ttl=PRE_TOURNEY_TTL_SEC,
        lock_key=f"pre:{tour}",
    )
