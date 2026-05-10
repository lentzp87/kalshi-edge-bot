"""Kalshi WebSocket client (skeleton).

Currently READ-ONLY: subscribes to ticker updates for a list of tickers
and dispatches to registered callbacks. Doesn't yet integrate with the
in-game model -- that's a follow-up. Goal here is to prove the WS pipe
works in production so we can iterate on it.

Graceful degradation:
  * If `websockets` library isn't installed -> log warning, return.
  * If WebSocket fails to connect (auth, network) -> log warning,
    schedule reconnect with exponential backoff (capped 60s), keep trying.
  * Bot keeps using REST polling regardless.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Callable

import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

log = structlog.get_logger(__name__)


# Per Kalshi docs: prod WS endpoint. Only one URL because WS lives on prod
# infrastructure even if you point REST at demo.
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"

# Backoff bounds for reconnect.
_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 60.0


TickCallback = Callable[[str, float, dict], None]


class KalshiWebSocket:
    """Read-only Kalshi WebSocket subscriber.

    Wraps an existing KalshiClient for auth (we reuse its loaded private
    key + key id rather than re-loading from env, so the signing logic
    stays in one place).
    """

    def __init__(self, client: Any) -> None:
        """client: an existing KalshiClient instance for auth helpers."""
        self._client = client
        # Pull what we need off the client's public attributes. KalshiClient
        # exposes `key_id` directly; the private key is on `_private_key`.
        # We deliberately don't import or reuse `_sign` -- per spec we copy
        # the signing logic so the WS module is self-contained.
        self._key_id: str | None = getattr(client, "key_id", None)
        self._private_key = getattr(client, "_private_key", None)

        self._callbacks: list[TickCallback] = []
        self._tickers: set[str] = set()
        # Map subscription cmd id -> tickers requested, for debugging.
        self._cmd_id = 0
        self._ws = None  # active websocket connection, if any
        self._lock = asyncio.Lock()  # serializes subscribe/unsubscribe writes
        self._available = True  # flipped to False if `websockets` is missing
        self._connected = asyncio.Event()  # set when handshake + sub succeed

    # ------------------------------------------------------------------ public

    def on_tick(self, callback: TickCallback) -> None:
        """Register a callback fn(ticker, mid, raw_msg). Multiple callbacks ok."""
        self._callbacks.append(callback)

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to additional tickers. Idempotent.

        New tickers are remembered so that reconnects re-subscribe to the
        full set. If we're currently connected, send the subscribe frame
        for just the *new* tickers right now.
        """
        new = [t for t in tickers if t and t not in self._tickers]
        for t in new:
            self._tickers.add(t)
        if not new:
            return
        if self._ws is not None:
            await self._send_subscribe(new)

    async def unsubscribe(self, tickers: list[str]) -> None:
        gone = [t for t in tickers if t in self._tickers]
        for t in gone:
            self._tickers.discard(t)
        if not gone or self._ws is None:
            return
        async with self._lock:
            self._cmd_id += 1
            msg = {
                "id": self._cmd_id,
                "cmd": "unsubscribe",
                "params": {"market_tickers": gone},
            }
            try:
                await self._ws.send(json.dumps(msg))
            except Exception as e:  # noqa: BLE001 - ws library raises various
                log.warning("kalshi_ws.unsubscribe_failed", err=str(e))

    async def run(self) -> None:
        """Main run loop. Connects, subscribes, reads, dispatches.

        Reconnects on transient failure with exponential backoff (cap 60s).
        Exits cleanly on cancellation.
        """
        # Local import so missing-library is a graceful no-op rather than a
        # hard import error at module load time.
        try:
            import websockets  # noqa: F401
            from websockets.exceptions import ConnectionClosed
        except ImportError:
            self._available = False
            log.warning(
                "kalshi_ws.unavailable",
                note="websockets library not installed; WS disabled, REST polling continues",
            )
            return

        if self._private_key is None or not self._key_id:
            log.warning(
                "kalshi_ws.no_auth",
                note="missing key id or private key; cannot connect",
            )
            return

        backoff = _BACKOFF_INITIAL
        while True:
            try:
                headers = self._signed_headers()
                # `additional_headers` is the modern websockets kwarg
                # (>= 12.0); older versions used `extra_headers`. We try
                # the modern name first and fall back if needed.
                connect = self._connect(headers)
                async with connect as ws:
                    self._ws = ws
                    log.info("kalshi_ws.connected", url=WS_URL)
                    # Re-subscribe to everything we know about.
                    if self._tickers:
                        await self._send_subscribe(sorted(self._tickers))
                    self._connected.set()
                    backoff = _BACKOFF_INITIAL  # reset on healthy connect
                    await self._read_loop(ws)
            except asyncio.CancelledError:
                log.info("kalshi_ws.cancelled")
                raise
            except ConnectionClosed as e:
                log.warning("kalshi_ws.closed", code=getattr(e, "code", None), reason=str(e))
            except Exception as e:  # noqa: BLE001
                log.warning("kalshi_ws.error", err=str(e), backoff=backoff)
            finally:
                self._ws = None
                self._connected.clear()

            # Sleep with backoff, then retry. Cap at 60s.
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, _BACKOFF_MAX)

    # ----------------------------------------------------------------- private

    def _connect(self, headers: dict[str, str]):
        """Return an `async with`-able websocket connect context.

        Handles the `additional_headers` vs `extra_headers` kwarg rename
        between websockets versions.
        """
        import websockets

        try:
            return websockets.connect(WS_URL, additional_headers=headers, ping_interval=20)
        except TypeError:
            # Older websockets (<12) used extra_headers
            return websockets.connect(WS_URL, extra_headers=headers, ping_interval=20)

    def _signed_headers(self) -> dict[str, str]:
        """Build Kalshi auth headers for the WS handshake.

        Same RSA-PSS-SHA256 scheme as REST (see KalshiClient._sign). Copied
        rather than imported so this module doesn't depend on a private
        method of another class.
        """
        ts = str(int(time.time() * 1000))
        message = f"{ts}GET{WS_PATH}".encode()
        sig = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self._key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    async def _send_subscribe(self, tickers: list[str]) -> None:
        if not tickers or self._ws is None:
            return
        async with self._lock:
            self._cmd_id += 1
            msg = {
                "id": self._cmd_id,
                "cmd": "subscribe",
                "params": {
                    "channels": ["ticker_v2"],
                    "market_tickers": tickers,
                },
            }
            try:
                await self._ws.send(json.dumps(msg))
                log.info("kalshi_ws.subscribe_sent", count=len(tickers), id=self._cmd_id)
            except Exception as e:  # noqa: BLE001
                log.warning("kalshi_ws.subscribe_failed", err=str(e))

    async def _read_loop(self, ws) -> None:
        """Read frames forever; dispatch ticker_v2 to callbacks."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                log.debug("kalshi_ws.bad_json", raw=str(raw)[:200])
                continue

            mtype = msg.get("type") or msg.get("channel")
            if mtype == "ticker_v2":
                self._dispatch_tick(msg)
            elif mtype in ("error", "subscribed", "unsubscribed", "ok"):
                # Ack / control frames -- log at debug.
                log.debug("kalshi_ws.control", type=mtype, msg=msg)
            else:
                # Unknown frame type -- log once at debug for visibility.
                log.debug("kalshi_ws.unknown_frame", type=mtype)

    def _dispatch_tick(self, msg: dict) -> None:
        """Compute mid, log smoke-test line, fan out to callbacks."""
        body = msg.get("msg") or msg  # tolerate flat or nested
        ticker = body.get("market_ticker") or body.get("ticker") or ""
        if not ticker:
            return

        # Field names per spec; fall back to generic names if Kalshi
        # ships a slightly different schema.
        try:
            yes_bid = float(body.get("yes_bid") or 0)
        except (TypeError, ValueError):
            yes_bid = 0.0
        try:
            yes_ask = float(body.get("yes_ask") or 0)
        except (TypeError, ValueError):
            yes_ask = 0.0
        try:
            price = float(body.get("price") or body.get("last_price") or 0)
        except (TypeError, ValueError):
            price = 0.0

        if yes_bid > 0 and yes_ask > 0:
            mid = (yes_bid + yes_ask) / 2.0
        else:
            mid = price

        log.info("kalshi_ws.tick", ticker=ticker, mid=mid)

        for cb in self._callbacks:
            try:
                cb(ticker, mid, msg)
            except Exception as e:  # noqa: BLE001 - one bad cb mustn't kill loop
                log.warning("kalshi_ws.callback_error", err=str(e))
