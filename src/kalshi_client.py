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
    # Per docs.kalshi.com: trading-api.kalshi.com is the canonical prod URL.
    # api.elections.kalshi.com is the legacy URL that still serves the same API.
    "prod": "https://trading-api.kalshi.com",
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
    def mid(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def spread_cents(self) -> int:
        return int(round((self.yes_ask - self.yes_bid) * 100))


class KalshiClient:
    def __init__(self) -> None:
        env = env_config()
        self.base_url = BASE_URLS[env.kalshi_env]
        self.key_id = env.kalshi_api_key_id
        self._private_key = self._load_private_key(env)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=15.0)

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
    ) -> tuple[list[Market], str | None]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = await self._request("GET", "/trade-api/v2/markets", params=params)
        raw_markets = data.get("markets", [])
        # Diagnostic: log response shape so we can see what Kalshi actually sent
        log.info(
            "kalshi.list_markets",
            base_url=self.base_url,
            status_filter=status,
            response_keys=list(data.keys()),
            num_markets=len(raw_markets),
            first_market_keys=list(raw_markets[0].keys()) if raw_markets else None,
        )
        markets = [self._parse_market(m) for m in raw_markets]
        return markets, data.get("cursor")

    async def get_orderbook(self, ticker: str) -> dict:
        return await self._request(
            "GET", f"/trade-api/v2/markets/{ticker}/orderbook"
        )

    @staticmethod
    def _parse_market(m: dict) -> Market:
        # Kalshi returns prices in cents (int). Normalize to dollars.
        cents_or_zero = lambda k: (m.get(k) or 0) / 100
        return Market(
            ticker=m["ticker"],
            title=m.get("title", ""),
            category=m.get("category", "Unknown"),
            yes_bid=cents_or_zero("yes_bid"),
            yes_ask=cents_or_zero("yes_ask"),
            last_price=cents_or_zero("last_price"),
            volume=m.get("volume", 0),
            open_interest=m.get("open_interest", 0),
            close_time_iso=m.get("close_time", ""),
            raw=m,
        )

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
