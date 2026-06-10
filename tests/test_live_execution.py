"""Unit tests for the live execution layer (entry fill reconciliation +
live sell path) added 2026-06-10. Run from repo root:

    python3 -m pytest tests/test_live_execution.py -q
or
    python3 tests/test_live_execution.py

Everything is mocked — no network, no real orders.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import ExecutionConfig                     # noqa: E402
from src.decision import TradeSignal                       # noqa: E402
from src.execution import Executor, OpenPosition           # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeClient:
    """Scriptable Kalshi client. Configure per-test via attributes."""

    def __init__(self):
        self.placed = []          # every place_order call body
        self.canceled = []        # order_ids
        self.order_status = "executed"
        self.fills_by_order = {}  # order_id -> list[fill dict]
        self.orderbook = {}
        self._oid = 0

    async def place_order(self, **kw):
        self._oid += 1
        oid = f"ord-{self._oid}"
        self.placed.append({**kw, "order_id": oid})
        return {"order": {"order_id": oid, "status": "resting"}}

    async def get_order(self, order_id):
        return {"order": {"order_id": order_id, "status": self.order_status}}

    async def get_fills(self, *, order_id=None, ticker=None, limit=100):
        return {"fills": self.fills_by_order.get(order_id, [])}

    async def cancel_order(self, order_id):
        self.canceled.append(order_id)
        return {}

    async def get_orderbook(self, ticker):
        return self.orderbook


class FakeRisk:
    def __init__(self):
        self.opens, self.closes = [], []

    def record_open(self, signal):
        self.opens.append(signal)

    def record_close(self, **kw):
        self.closes.append(kw)


class FakeJournal:
    def __init__(self):
        self.opened, self.closed = [], []

    def log_open(self, **kw):
        self.opened.append(kw)

    def log_close(self, **kw):
        self.closed.append(kw)


class FakeMarket:
    ticker = "KXATPMATCH-26JUN10TEST-AAA"
    raw = {}


def make_signal(side="yes", price_cents=55):
    return TradeSignal(
        ticker=FakeMarket.ticker, side=side, price_cents=price_cents,
        size_usd=2.0, edge=0.08, reason="test",
    )


def make_executor(mode="live"):
    ex = Executor.__new__(Executor)  # skip __init__ (reads config/env)
    ex.client = FakeClient()
    ex.risk = FakeRisk()
    ex.journal = FakeJournal()
    ex.mode = mode
    ex.cfg = ExecutionConfig(
        take_profit_pct=0.50, stop_loss_pct=1.00,
        scale_in_chunks=1,
        live_entry_fill_timeout_s=1, live_fill_poll_s=0,
        live_exit_reprice_s=0, live_exit_max_reprices=2,
        live_exit_retry_cooldown_s=0,
    )
    ex.open = {}
    ex.open_events = set()
    ex.cooldown_until = {}
    ex.whale_tracker = None
    ex.revalidate_edge = None
    return ex


def make_pos(ex, side="yes", fill=0.40, contracts=3):
    pos = OpenPosition(
        signal=make_signal(side=side), fill_price=fill,
        contracts=contracts, opened_at=datetime.now(timezone.utc),
    )
    ex.open[FakeMarket.ticker] = pos
    return pos


# ---------------------------------------------------------------------------
# Entry reconciliation
# ---------------------------------------------------------------------------

def test_entry_full_fill():
    ex = make_executor()
    ex.client.fills_by_order["ord-1"] = [
        {"count": 2, "yes_price": 54, "side": "yes", "action": "buy"},
        {"count": 1, "yes_price": 55, "side": "yes", "action": "buy"},
    ]
    asyncio.run(ex._submit_live(make_signal(), FakeMarket(), 3))
    assert len(ex.journal.opened) == 1, "should journal exactly one open"
    row = ex.journal.opened[0]
    assert row["contracts"] == 3
    expected = (2 * 0.54 + 1 * 0.55) / 3
    assert abs(row["fill_price"] - expected) < 1e-9, \
        f"journal must record ACTUAL avg fill, got {row['fill_price']}"
    print("entry full fill: OK")


def test_entry_zero_fill():
    ex = make_executor()
    ex.client.order_status = "canceled"
    # no fills configured -> nothing executed
    asyncio.run(ex._submit_live(make_signal(), FakeMarket(), 3))
    assert ex.journal.opened == [], "zero fills must journal NOTHING"
    assert ex.client.canceled, "unfilled orders must be canceled"
    assert FakeMarket.ticker not in ex.open
    print("entry zero fill: OK")


def test_entry_partial_fill():
    ex = make_executor()
    ex.client.fills_by_order["ord-1"] = [
        {"count": 1, "yes_price": 55, "side": "yes", "action": "buy"},
    ]
    asyncio.run(ex._submit_live(make_signal(), FakeMarket(), 3))
    row = ex.journal.opened[0]
    assert row["contracts"] == 1, "journal the REAL count, not requested"
    assert abs(row["fill_price"] - 0.55) < 1e-9
    print("entry partial fill: OK")


# ---------------------------------------------------------------------------
# Live exit
# ---------------------------------------------------------------------------

def test_exit_live_full_fill():
    ex = make_executor()
    pos = make_pos(ex, contracts=3)
    ex.client.orderbook = {"orderbook_fp": {
        "yes_dollars": [["0.5800", "10.00"], ["0.6000", "5.00"]],
        "no_dollars": [["0.3800", "10.00"]],
    }}

    def fills_after_place():
        # first sell order gets fully filled at 60c
        ex.client.fills_by_order[f"ord-{ex.client._oid}"] = [
            {"count": 3, "yes_price": 60, "side": "yes", "action": "sell"},
        ]
    orig = ex.client.place_order

    async def place_and_fill(**kw):
        r = await orig(**kw)
        fills_after_place()
        return r
    ex.client.place_order = place_and_fill

    closed = asyncio.run(ex._exit(FakeMarket.ticker, 0.61,
                                  reason="take_profit"))
    assert closed is True
    assert FakeMarket.ticker not in ex.open
    row = ex.journal.closed[0]
    assert abs(row["exit_price"] - 0.60) < 1e-9, \
        "must journal the REAL fill (0.60), not the trigger mid (0.61)"
    # sell order was placed at the best bid (60c, max of the stack)
    sell = ex.client.placed[0]
    assert sell["action"] == "sell" and sell["price_cents"] == 60
    print("live exit full fill: OK")


def test_exit_live_no_bids_keeps_position():
    ex = make_executor()
    make_pos(ex)
    ex.client.orderbook = {"orderbook_fp": {
        "yes_dollars": [], "no_dollars": [["0.3800", "10.00"]],
    }}
    closed = asyncio.run(ex._exit(FakeMarket.ticker, 0.61,
                                  reason="take_profit"))
    assert closed is False, "no bids -> exit must fail"
    assert FakeMarket.ticker in ex.open, "position must REMAIN OPEN"
    assert ex.journal.closed == [], "nothing may be journaled"
    assert all(p["action"] != "sell" for p in ex.client.placed)
    print("live exit no bids: OK")


def test_exit_live_partial_accumulates():
    ex = make_executor()
    pos = make_pos(ex, contracts=4)
    ex.client.orderbook = {"orderbook_fp": {
        "yes_dollars": [["0.6000", "10.00"]], "no_dollars": [],
    }}
    fill_script = [2, 0, 0]  # first order fills 2 of 4, rest nothing

    orig = ex.client.place_order

    async def place_scripted(**kw):
        r = await orig(**kw)
        n = fill_script.pop(0) if fill_script else 0
        if n:
            ex.client.fills_by_order[f"ord-{ex.client._oid}"] = [
                {"count": n, "yes_price": 60, "side": "yes",
                 "action": "sell"},
            ]
        return r
    ex.client.place_order = place_scripted

    closed = asyncio.run(ex._exit(FakeMarket.ticker, 0.61, reason="time_exit"))
    assert closed is False, "partial fill -> not closed yet"
    assert FakeMarket.ticker in ex.open
    assert getattr(pos, "sold_contracts", 0) == 2, "partial must accumulate"

    # next watcher tick: remaining 2 fill
    fill_script.extend([2])
    closed = asyncio.run(ex._exit(FakeMarket.ticker, 0.61, reason="time_exit"))
    assert closed is True
    assert abs(ex.journal.closed[0]["exit_price"] - 0.60) < 1e-9
    print("live exit partial accumulation: OK")


def test_exit_paper_unchanged():
    ex = make_executor(mode="paper")
    make_pos(ex, fill=0.40, contracts=3)
    closed = asyncio.run(ex._exit(FakeMarket.ticker, 0.61,
                                  reason="take_profit"))
    assert closed is True
    assert abs(ex.journal.closed[0]["exit_price"] - 0.61) < 1e-9, \
        "paper mode still books the optimistic mid"
    print("paper exit unchanged: OK")


if __name__ == "__main__":
    test_entry_full_fill()
    test_entry_zero_fill()
    test_entry_partial_fill()
    test_exit_live_full_fill()
    test_exit_live_no_bids_keeps_position()
    test_exit_live_partial_accumulates()
    test_exit_paper_unchanged()
    print("\nall live-execution tests passed")
