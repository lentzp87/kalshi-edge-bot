"""Thin async wrapper around the Kalshi v2 REST API.

Auth uses the RSA-signed-header scheme that Kalshi's public API requires:
  KALSHI-ACCESS-KEY        = your key id
  KALSHI-ACCESS-TIMESTAMP  = unix ms
  KALSHI-ACCESS-SIGNATURE  = base64(RSA-PSS-SHA256(timestamp + method + path))

Endpoints implemented are the minimum needed to scan markets and trade:
  GET  /trade-api/v2/markets
  GET  /trade-api/v2/markets/{ticker}
  GET  /trade-api/v2/markets/{ticker}/orderbook
  GET  /trade-api/v2/portfolio/positions
  POST /trade-api/v2/portfolio/orders
  DELETE /trade-api/v2/portfolio/orders/{order_id}

If you switch from the demo to the prod env later, only the base URL changes.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .config import env_config

log = structlog.get_logger(__name__)

BASE_URLS = {
    "demo": "https://demo-api.kalshi.co",
    # Confirmed by Kalshi's own redirect message: trading-api.kalshi.com
    # returns 401 "API has been moved to https://api.elections.kalshi.com/".
    "prod": "https://api.elections.kalshi.com",
}


@dataclass
class Market:
    ticker: str
    title: str
    category: str         # "Weather", "Economics", "Sports", "Crypto", "Politics", ...
    yes_bid: float        # in dollars (0.0 - 1.0)
    yes_ask: float
    last_price: float
    volume: int
    open_interest: int
    close_time_iso: str   # ISO8601
    raw: dict             # original payload, for model code

    @property
    def effective_yes_bid(self) -> float:
        """Best YES-equivalent bid: max of (direct yes_bid, 1 - no_ask).

        On Kalshi, buying NO at price p is the same as selling YES at price (1-p).
        So if nobody is directly bidding YES but somebody is asking NO, the implied
        YES bid is (1 - no_ask). This unifies the two sides of the book.
        """
        no_ask = float(self.raw.get("no_ask_dollars") or 0)
        implied_from_no = (1.0 - no_ask) if no_ask > 0 else 0.0
        return max(self.yes_bid or 0.0, implied_from_no)

    @property
    def effective_yes_ask(self) -> float:
        """Best YES-equivalent ask: min of (direct yes_ask, 1 - no_bid)."""
        no_bid = float(self.raw.get("no_bid_dollars") or 0)
        implied_from_no = (1.0 - no_bid) if no_bid > 0 else 0.0
        candidates = [v for v in (self.yes_ask or 0.0, implied_from_no) if v > 0]
        return min(candidates) if candidates else 0.0

    @property
    def mid(self) -> float:
        bid = self.effective_yes_bid
        ask = self.effective_yes_ask
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return self.last_price

    @property
    def spread_cents(self) -> int:
        bid = self.effective_yes_bid
        ask = self.effective_yes_ask
        if bid > 0 and ask > 0:
            return int(round((ask - bid) * 100))
        return 100  # treat one-sided book as huge spread = reject


class KalshiClient:
    def __init__(self) -> None:
        env = env_config()
        self.base_url = BASE_URLS[env.kalshi_env]
        self.key_id = env.kalshi_api_key_id
        self._private_key = self._load_private_key(env)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=15.0)
        # Populated by load_series_categories() at startup; maps each
        # Kalshi series_ticker (e.g. "KXWTAMATCH") to its real category
        # ("Sports", "Climate and Weather", etc.) per Kalshi's own taxonomy.
        self.series_categories: dict[str, str] = {}

    @staticmethod
    def _load_private_key(env) -> Any:
        """Load PEM from env var or file.

        Render env vars often arrive with literal '\\n' instead of real
        newlines, or wrapped in surrounding quotes. We normalize both.
        """
        if env.kalshi_api_private_key:
            raw = env.kalshi_api_private_key.strip()
            if raw.startswith(('"', "'")) and raw.endswith(('"', "'")):
                raw = raw[1:-1]
            # Convert literal "\n" sequences into real newlines
            if "\\n" in raw and "\n" not in raw:
                raw = raw.replace("\\n", "\n")
            pem_bytes = raw.encode()
        elif env.kalshi_api_private_key_path:
            pem_bytes = Path(env.kalshi_api_private_key_path).read_bytes()
        else:
            log.warning("kalshi.no_private_key", note="signed calls will fail")
            return None
        try:
            return serialization.load_pem_private_key(pem_bytes, password=None)
        except Exception as e:
            log.error("kalshi.bad_private_key", err=str(e))
            return None

    def _sign(self, method: str, path: str) -> dict[str, str]:
        if self._private_key is None:
            return {}
        ts = str(int(time.time() * 1000))
        message = f"{ts}{method.upper()}{path}".encode()
        sig = self._private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    @retry(
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _request(self, method: str, path: str, **kwargs) -> dict:
        headers = self._sign(method, path)
        headers.update(kwargs.pop("headers", {}))
        r = await self._client.request(method, path, headers=headers, **kwargs)
        if r.status_code >= 400:
            log.error(
                "kalshi.http_error",
                method=method,
                path=path,
                status=r.status_code,
                body=r.text[:500],
            )
        r.raise_for_status()
        return r.json()

    # ---------- public market data ----------

    async def list_markets(
        self,
        *,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
        series_ticker: str | None = None,
    ) -> tuple[list[Market], str | None]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = await self._request("GET", "/trade-api/v2/markets", params=params)
        raw_markets = data.get("markets", [])
        # Quiet per-page log — just the count, no field-name dump.
        log.debug("kalshi.list_markets.page", num_markets=len(raw_markets))
        markets = [self._parse_market(m) for m in raw_markets]
        return markets, data.get("cursor")

    async def get_orderbook(self, ticker: str) -> dict:
        return await self._request(
            "GET", f"/trade-api/v2/markets/{ticker}/orderbook"
        )

    async def list_series(self) -> list[dict]:
        """Fetch the full series catalog. Public endpoint, no auth needed."""
        data = await self._request("GET", "/trade-api/v2/series")
        return data.get("series", [])

    async def load_series_categories(self) -> int:
        """Populate self.series_categories from Kalshi's series catalog.

        Call once at startup. Returns the number of series loaded.
        """
        try:
            series = await self.list_series()
        except Exception as e:
            log.exception("kalshi.load_series_failed", err=str(e))
            return 0
        self.series_categories = {
            s["ticker"]: s.get("category", "Unknown")
            for s in series if s.get("ticker")
        }
        log.info("kalshi.series_loaded", count=len(self.series_categories))
        return len(self.series_categories)

    def category_for_event(self, event_ticker: str) -> str:
        """Look up category for a market's event_ticker.

        Kalshi event tickers look like 'KXWTAMATCH-26MAY04TOWSRA' — the part
        before the first '-' is the series_ticker, which we map via the
        catalog loaded at startup.
        """
        if not event_ticker:
            return "Unknown"
        series = event_ticker.split("-", 1)[0]
        return self.series_categories.get(series) or _category_from_event_ticker(event_ticker)

    def _parse_market(self, m: dict) -> Market:
        """Parse a Kalshi market into our normalized Market dataclass.

        Kalshi's current schema uses dollar-denominated price fields with the
        `_dollars` suffix (already 0.0-1.0) and float-typed sizes with `_fp`.
        We fall back to legacy cents-based field names for any older response
        shape, so this works on both demo and prod.
        """
        def dollars_or_legacy(dollar_key: str, legacy_cents_key: str) -> float:
            v = m.get(dollar_key)
            if v is not None:
                return float(v)
            return (m.get(legacy_cents_key) or 0) / 100

        return Market(
            ticker=m["ticker"],
            title=m.get("title", ""),
            # Kalshi no longer returns `category` per market — we derive it
            # via the series_ticker prefix using the catalog loaded at startup.
            category=self.category_for_event(m.get("event_ticker", "")),
            yes_bid=dollars_or_legacy("yes_bid_dollars", "yes_bid"),
            yes_ask=dollars_or_legacy("yes_ask_dollars", "yes_ask"),
            last_price=dollars_or_legacy("last_price_dollars", "last_price"),
            volume=int(float(m.get("volume_24h_fp") or m.get("volume_fp") or m.get("volume") or 0)),
            open_interest=int(float(m.get("open_interest_fp") or m.get("open_interest") or 0)),
            # Use the actual *resolution* time (when the event happens / market
            # settles), not close_time which is when *trading* stops — those
            # can differ by weeks for sports playoff markets.
            close_time_iso=(
                m.get("expected_expiration_time")
                or m.get("occurrence_datetime")
                or m.get("expiration_time")
                or m.get("close_time", "")
            ),
            raw=m,
        )


def _category_from_event_ticker(event_ticker: str) -> str:
    """Map Kalshi event-ticker prefixes to broad categories.

    This is a best-effort categorization so the model registry can route
    markets to the right model. Extend as you discover more series.
    """
    et = event_ticker.upper()
    if not et:
        return "Unknown"
    # Sports — Kalshi sports series start with KX + league code
    if any(et.startswith(p) for p in ("KXATP", "KXNFL", "KXNBA", "KXMLB", "KXNHL",
                                      "KXUFC", "KXBOX", "KXPGA", "KXTEN", "KXSOC",
                                      "KXNCAA", "KXWTA", "KXFIFA")):
        return "Sports"
    # Weather — temp / rain / snow series
    if any(et.startswith(p) for p in ("KXHIGH", "KXLOW", "KXRAIN", "KXSNOW",
                                      "KXTEMP", "KXHOT", "KXCOLD", "KXHURR")):
        return "Weather"
    # Crypto
    if any(et.startswith(p) for p in ("KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXCRYPTO")):
        return "Crypto"
    # Economics / Fed / inflation
    if any(et.startswith(p) for p in ("KXFED", "KXCPI", "KXJOBS", "KXGDP",
                                      "KXNFP", "KXUNRATE", "KXPCE")):
        return "Economics"
    # Politics
    if any(et.startswith(p) for p in ("KXPRES", "KXSEN", "KXHOUSE", "KXGOV",
                                      "KXELECT", "KXSCOTUS", "KXIMP")):
        return "Politics"
    return "Unknown"

    # ---------- portfolio + orders ----------

    async def get_positions(self) -> dict:
        return await self._request("GET", "/trade-api/v2/portfolio/positions")

    async def place_order(
        self,
        *,
        ticker: str,
        side: str,        # "yes" or "no"
        action: str,      # "buy" or "sell"
        count: int,
        price_cents: int,
        client_order_id: str,
    ) -> dict:
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "yes_price": price_cents if side == "yes" else None,
            "no_price": price_cents if side == "no" else None,
        }
        body = {k: v for k, v in body.items() if v is not None}
        return await self._request("POST", "/trade-api/v2/portfolio/orders", json=body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self._request(
            "DELETE", f"/trade-api/v2/portfolio/orders/{order_id}"
        )

    async def aclose(self) -> None:
        await self._client.aclose()
