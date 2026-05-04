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
            "wrong_category": 0,
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
        # Category whitelist (cheapest filter — do it first)
        if self.cfg.allowed_categories and m.category not in self.cfg.allowed_categories:
            self.rejection_counts["wrong_category"] += 1
            return False

        # Price band — skip extreme tails where reward/risk is bad
        if not (self.cfg.price_min <= m.mid <= self.cfg.price_max):
            self.rejection_counts["price_band"] += 1
            return False

        # Spread filter
        if m.spread_cents > self.cfg.max_spread_cents:
            self.rejection_counts["spread"] += 1
            return False

        # Liquidity: compute from order-book depth (Kalshi's liquidity_dollars
        # field returns 0 even for markets with real resting orders, so we
        # ignore it). Approximate dollar liquidity = bid_size*bid + ask_size*ask.
        bid_size = float(m.raw.get("yes_bid_size_fp") or 0)
        ask_size = float(m.raw.get("yes_ask_size_fp") or 0)
        # Kalshi contracts settle at $1, so size is in contracts. Dollar value
        # of resting orders is roughly: bid_size * bid_price + ask_size * ask_price.
        liq_usd = (bid_size * m.yes_bid) + (ask_size * m.yes_ask)
        if liq_usd < self.cfg.min_liquidity_usd:
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
