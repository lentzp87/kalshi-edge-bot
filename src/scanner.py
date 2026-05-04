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
        # Per-iteration counters; reset at the start of each scan.
        self.rejection_counts: dict[str, int] = {}

    async def stream_tradeable_markets(self) -> AsyncIterator[Market]:
        # Reset counters for this scan
        self.rejection_counts = {
            "total_seen": 0,
            "price_band": 0,
            "spread": 0,
            "liquidity": 0,
            "expiry_too_soon": 0,
            "expiry_too_far": 0,
            "expiry_unparseable": 0,
        }
        # If specific series are configured, scan only those (fast + targeted).
        # Otherwise walk the full catalog up to max_pages_per_scan.
        if self.cfg.series_tickers:
            for series in self.cfg.series_tickers:
                async for m in self._stream_series(series):
                    yield m
        else:
            async for m in self._stream_all():
                yield m

    async def _stream_series(self, series_ticker: str) -> AsyncIterator[Market]:
        cursor: str | None = None
        page = 0
        while True:
            page += 1
            markets, cursor = await self.client.list_markets(
                cursor=cursor, series_ticker=series_ticker
            )
            log.info("scanner.series_page", series=series_ticker, page=page, fetched=len(markets))
            for m in markets:
                self.rejection_counts["total_seen"] += 1
                if self._is_tradeable(m):
                    yield m
            if not cursor:
                break
            if page >= self.cfg.max_pages_per_scan:
                log.info("scanner.page_cap_hit", series=series_ticker, page=page,
                         cap=self.cfg.max_pages_per_scan)
                break

    async def _stream_all(self) -> AsyncIterator[Market]:
        cursor: str | None = None
        page = 0
        while True:
            page += 1
            markets, cursor = await self.client.list_markets(cursor=cursor)
            log.info("scanner.page", page=page, fetched=len(markets))
            for m in markets:
                self.rejection_counts["total_seen"] += 1
                if self._is_tradeable(m):
                    yield m
            if not cursor:
                break
            if page >= self.cfg.max_pages_per_scan:
                log.info("scanner.page_cap_hit", page=page, cap=self.cfg.max_pages_per_scan)
                break

    def _is_tradeable(self, m: Market) -> bool:
        # Price band — skip extreme tails where reward/risk is bad
        if not (self.cfg.price_min <= m.mid <= self.cfg.price_max):
            self.rejection_counts["price_band"] += 1
            return False

        # Spread filter
        if m.spread_cents > self.cfg.max_spread_cents:
            self.rejection_counts["spread"] += 1
            return False

        # Liquidity: prefer Kalshi's direct `liquidity_dollars` if present;
        # fall back to (open_interest * mid) as a proxy for older payloads.
        liq_dollars = m.raw.get("liquidity_dollars")
        approx_liq_usd = float(liq_dollars) if liq_dollars is not None else m.open_interest * m.mid
        if approx_liq_usd < self.cfg.min_liquidity_usd:
            self.rejection_counts["liquidity"] += 1
            return False

        # Time-to-expiry window
        if m.close_time_iso:
            try:
                close_dt = datetime.fromisoformat(m.close_time_iso.replace("Z", "+00:00"))
                minutes_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60
                if minutes_left < self.cfg.min_minutes_to_expiry:
                    self.rejection_counts["expiry_too_soon"] += 1
                    return False
                if minutes_left > self.cfg.max_minutes_to_expiry:
                    self.rejection_counts["expiry_too_far"] += 1
                    return False
            except ValueError:
                self.rejection_counts["expiry_unparseable"] += 1
                return False

        return True
