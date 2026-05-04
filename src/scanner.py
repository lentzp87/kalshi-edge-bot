"""Scanner: pull all open markets and filter to a tradeable subset."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

import structlog

from .config import file_config
from .kalshi_client import KalshiClient, Market

log = structlog.get_logger(__name__)


class Scanner:
    def __init__(self, client: KalshiClient) -> None:
        self.client = client
        self.cfg = file_config().scanner

    async def stream_tradeable_markets(self) -> AsyncIterator[Market]:
        cursor: str | None = None
        while True:
            markets, cursor = await self.client.list_markets(cursor=cursor)
            for m in markets:
                if self._is_tradeable(m):
                    yield m
            if not cursor:
                break

    def _is_tradeable(self, m: Market) -> bool:
        # Price band — skip extreme tails where reward/risk is bad
        if not (self.cfg.price_min <= m.mid <= self.cfg.price_max):
            return False

        # Spread filter
        if m.spread_cents > self.cfg.max_spread_cents:
            return False

        # Liquidity: prefer Kalshi's direct `liquidity_dollars` if present;
        # fall back to (open_interest * mid) as a proxy for older payloads.
        liq_dollars = m.raw.get("liquidity_dollars")
        approx_liq_usd = float(liq_dollars) if liq_dollars is not None else m.open_interest * m.mid
        if approx_liq_usd < self.cfg.min_liquidity_usd:
            return False

        # Time-to-expiry window
        if m.close_time_iso:
            try:
                close_dt = datetime.fromisoformat(m.close_time_iso.replace("Z", "+00:00"))
                minutes_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60
                if minutes_left < self.cfg.min_minutes_to_expiry:
                    return False
                if minutes_left > self.cfg.max_minutes_to_expiry:
                    return False
            except ValueError:
                return False

        return True
