"""Polymarket Gamma API client (read-only).

Pulls active markets from the public Gamma endpoint. No auth is needed.
We keep a single global cache (5 min TTL) shared across all callers, with
an asyncio lock so concurrent refreshes don't multiply the network cost.

Usage:
    from .polymarket_client import list_active_markets, parse_yes_price

    markets = await list_active_markets()
    for m in markets:
        p = parse_yes_price(m)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

BASE = "https://gamma-api.polymarket.com"
CACHE_TTL_SEC = 300  # 5 min

# (timestamp, list[dict]) — module-level so all callers share it.
_market_cache: tuple[float, list[dict]] | None = None
_lock = asyncio.Lock()


async def list_active_markets(
    max_pages: int = 5,
    page_size: int = 100,
) -> list[dict]:
    """Fetches active, non-closed markets. Cached 5 min globally.

    Returns up to max_pages * page_size markets. On any network or
    parsing error we log a warning and return the previously cached
    list (or [] if there's no cache yet).
    """
    global _market_cache

    async with _lock:
        now = time.time()
        if _market_cache is not None:
            ts, cached = _market_cache
            if now - ts < CACHE_TTL_SEC:
                return cached

        all_markets: list[dict] = []
        try:
            async with httpx.AsyncClient(base_url=BASE, timeout=15.0) as client:
                for page in range(max_pages):
                    offset = page * page_size
                    params = {
                        "active": "true",
                        "closed": "false",
                        "limit": page_size,
                        "offset": offset,
                    }
                    r = await client.get("/markets", params=params)
                    r.raise_for_status()
                    payload = r.json()
                    # Gamma returns a bare JSON array.
                    if isinstance(payload, list):
                        batch = payload
                    elif isinstance(payload, dict):
                        # Defensive: some deployments wrap in {"data": [...]}.
                        batch = payload.get("data") or payload.get("markets") or []
                    else:
                        batch = []
                    if not batch:
                        break
                    all_markets.extend(batch)
                    # Stop early if the API returned a short page.
                    if len(batch) < page_size:
                        break
        except Exception as e:
            log.warning("polymarket.fetch_failed", err=str(e))
            # On failure prefer the previously cached list over an empty
            # response — stale data beats no data for spread monitoring.
            if _market_cache is not None:
                return _market_cache[1]
            return []

        _market_cache = (now, all_markets)
        log.info(
            "polymarket.markets_fetched",
            count=len(all_markets),
            pages=max_pages,
        )
        return all_markets


def _maybe_json_loads(x: Any) -> Any:
    """outcomes / outcomePrices are JSON-string-encoded lists in Gamma,
    but defensive code: handle the case where they're already parsed."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (ValueError, json.JSONDecodeError):
            return None
    return x


def parse_yes_price(market: dict) -> float | None:
    """Pull the 'Yes' side probability from a market dict.

    outcomes / outcomePrices are JSON-string-encoded lists. The 'Yes'
    side's probability is outcomePrices[i] where outcomes[i] == 'Yes'.
    Returns None if it can't parse.
    """
    if not market:
        return None
    outcomes = _maybe_json_loads(market.get("outcomes"))
    prices = _maybe_json_loads(market.get("outcomePrices"))
    if not isinstance(outcomes, list) or not isinstance(prices, list):
        return None
    if len(outcomes) != len(prices):
        return None
    for outcome, price in zip(outcomes, prices):
        if not isinstance(outcome, str):
            continue
        if outcome.strip().lower() == "yes":
            try:
                return float(price)
            except (TypeError, ValueError):
                return None
    return None
