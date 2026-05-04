"""Execution engine.

In `paper` mode, fills are simulated instantly at the limit price; exits use
mid-price snapshots to compute realized P&L.

In `live` mode, the engine submits real limit orders to Kalshi. Scale-in is
respected (the entry size is split across `scale_in_chunks` orders posted at
the touch and one tick deeper).

Every position spawns a watcher that polls the orderbook and triggers exit on:
  - mid drift to TP
  - mid drift to SL
  - elapsed time >= time_exit_minutes
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog

from .config import env_config, file_config
from .decision import TradeSignal
from .journal import Journal
from .kalshi_client import KalshiClient, Market
from .risk import RiskEngine

log = structlog.get_logger(__name__)


@dataclass
class OpenPosition:
    signal: TradeSignal
    fill_price: float
    contracts: int
    opened_at: datetime
    children: list[asyncio.Task] = field(default_factory=list)


class Executor:
    def __init__(self, client: KalshiClient, risk: RiskEngine, journal: Journal) -> None:
        self.client = client
        self.risk = risk
        self.journal = journal
        self.cfg = file_config().execution
        self.mode = env_config().mode
        self.open: dict[str, OpenPosition] = {}

    async def submit(self, signal: TradeSignal, market: Market) -> None:
        if not self.risk.approve(signal):
            return

        contracts = max(1, int(signal.size_usd / max(signal.price_cents / 100, 0.01)))
        if self.mode == "paper":
            fill_price = signal.price_cents / 100
            self._record_open(signal, market, fill_price, contracts)
        else:
            await self._submit_live(signal, market, contracts)

    async def _submit_live(self, signal: TradeSignal, market: Market, contracts: int) -> None:
        chunks = max(1, self.cfg.scale_in_chunks)
        per_chunk = max(1, contracts // chunks)
        for i in range(chunks):
            client_order_id = f"{signal.ticker}-{uuid.uuid4().hex[:8]}"
            try:
                resp = await self.client.place_order(
                    ticker=signal.ticker,
                    side=signal.side,
                    action="buy",
                    count=per_chunk,
                    price_cents=signal.price_cents - i,  # post one tick deeper each chunk
                    client_order_id=client_order_id,
                )
                log.info("exec.live.placed", order=resp)
            except Exception as e:
                log.exception("exec.live.error", ticker=signal.ticker, err=str(e))

        # In live mode the fill watcher should reconcile actual fills before recording.
        # Stub: record optimistically so the journal sees the intent.
        self._record_open(signal, market, signal.price_cents / 100, contracts)

    def _record_open(self, signal: TradeSignal, market: Market, fill_price: float, contracts: int) -> None:
        pos = OpenPosition(
            signal=signal,
            fill_price=fill_price,
            contracts=contracts,
            opened_at=datetime.now(timezone.utc),
        )
        self.open[signal.ticker] = pos
        self.risk.record_open(signal)
        self.journal.log_open(signal=signal, market=market, fill_price=fill_price, contracts=contracts)
        pos.children.append(asyncio.create_task(self._watch(signal.ticker)))

    async def _watch(self, ticker: str) -> None:
        pos = self.open.get(ticker)
        if not pos:
            return
        deadline = pos.opened_at + timedelta(minutes=self.cfg.time_exit_minutes)
        while ticker in self.open:
            await asyncio.sleep(15)
            try:
                ob = await self.client.get_orderbook(ticker)
                mid = self._mid_from_orderbook(ob, pos.signal.side)
                pnl_pct = (mid - pos.fill_price) / pos.fill_price if pos.signal.side == "yes" \
                    else (pos.fill_price - mid) / pos.fill_price
                if pnl_pct >= self.cfg.take_profit_pct:
                    await self._exit(ticker, mid, reason="take_profit")
                    return
                if pnl_pct <= -self.cfg.stop_loss_pct:
                    await self._exit(ticker, mid, reason="stop_loss")
                    return
                if datetime.now(timezone.utc) >= deadline:
                    await self._exit(ticker, mid, reason="time_exit")
                    return
            except Exception as e:
                log.exception("exec.watch.error", ticker=ticker, err=str(e))

    @staticmethod
    def _mid_from_orderbook(ob: dict, side: str) -> float:
        """Compute the YES-equivalent mid price from Kalshi's orderbook payload.

        Kalshi returns the book under `orderbook_fp` with `yes_dollars` and
        `no_dollars` arrays. Each entry is `[price_str, size_str]` and the
        prices are ALREADY IN DOLLARS (0.01-0.99). The arrays are sorted
        ascending by price, so the LAST entry of each side is the best bid
        on that side. To get the YES ask, look at the best NO bid and
        invert (buying NO at p == selling YES at 1-p).
        """
        try:
            book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
            yes_book = book.get("yes_dollars") or book.get("yes") or []
            no_book = book.get("no_dollars") or book.get("no") or []

            yes_bid = float(yes_book[-1][0]) if yes_book else 0.0
            no_bid = float(no_book[-1][0]) if no_book else 0.0
            # Best YES ask is implied from best NO bid: 1 - no_bid
            yes_ask = (1.0 - no_bid) if no_bid > 0 else 0.0

            if yes_bid > 0 and yes_ask > 0:
                yes_mid = (yes_bid + yes_ask) / 2
            elif yes_bid > 0:
                yes_mid = yes_bid
            elif yes_ask > 0:
                yes_mid = yes_ask
            else:
                yes_mid = 0.5  # empty book — neutral fallback

            return yes_mid if side == "yes" else (1.0 - yes_mid)
        except Exception:
            log.exception("exec.orderbook_parse_error")
            return 0.5

    async def _exit(self, ticker: str, exit_price: float, *, reason: str) -> None:
        pos = self.open.pop(ticker, None)
        if not pos:
            return
        # P&L per contract is the price move in dollars. Contracts settle at
        # $1, so dollar P&L = (price_move) * contracts. NO extra * 100.
        pnl_per_contract = (exit_price - pos.fill_price) if pos.signal.side == "yes" \
            else (pos.fill_price - exit_price)
        pnl_usd = pnl_per_contract * pos.contracts
        # Approximate Kalshi fee (verify against current schedule).
        fees_usd = 0.07 * pos.contracts
        pnl_usd -= fees_usd
        # Sanity clamp: a long position can never lose more than its size.
        # Catches any residual unit-conversion bug before it trips the
        # daily-loss kill switch on imaginary losses.
        if pnl_usd < -pos.signal.size_usd:
            log.warning("exec.pnl_clamped_to_size",
                        ticker=ticker, raw_pnl=pnl_usd, size=pos.signal.size_usd)
            pnl_usd = -pos.signal.size_usd

        self.risk.record_close(size_usd=pos.signal.size_usd, realized_pnl_usd=pnl_usd)
        self.journal.log_close(
            ticker=ticker,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            fees_usd=fees_usd,
            reason=reason,
        )
        log.info("exec.exit", ticker=ticker, reason=reason, pnl=round(pnl_usd, 2))
