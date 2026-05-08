"""Settlement backtester.

For every closed trade in the journal, periodically poll Kalshi for the
market's resolution. Once Kalshi reports a `result`, we know which side
won — and we can compute the counterfactual P&L if we had held to
settlement instead of exiting via stop_loss / take_profit / time_exit.

Why this matters
----------------
Exit P&L on a single trade tells you whether *that* exit timing was
right. Held-to-settlement P&L tells you whether the *entry* itself was
+EV. These are different questions; both matter:

  * If actual_total < settlement_total → exits cost us money. Stops
    are likely too tight. Loosen SL or remove it.
  * If actual_total ≈ settlement_total → exits are neutral; entry is
    the limiting factor.
  * If actual_total > settlement_total → exits are saving us from
    bigger losses. Keep them.

The dashboard surfaces this comparison directly.
"""

from __future__ import annotations

import asyncio

import structlog

from .fee_model import fee_per_contract_dollars
from .journal import Journal
from .kalshi_client import KalshiClient

log = structlog.get_logger(__name__)


def _settlement_pnl(
    *, side: str, fill_price: float, contracts: int, our_side_won: bool,
) -> float:
    """P&L if we held the position to settlement.

    Each contract pays $1 if our side wins, $0 if it loses. We paid
    `fill_price` per contract on entry. So per-contract gross:
        win  → 1.0 - fill_price
        loss →     -fill_price

    Holding to settlement skips the exit fee entirely (Kalshi auto-
    settles), so we only subtract the entry fee.
    """
    per_contract = (1.0 - fill_price) if our_side_won else (-fill_price)
    gross = per_contract * contracts
    entry_fee = fee_per_contract_dollars(fill_price) * contracts
    return gross - entry_fee


async def _check_one(client: KalshiClient, journal: Journal, trade: dict) -> bool:
    """Poll Kalshi for one trade's settlement. Returns True on update."""
    ticker = trade["ticker"]
    try:
        market = await client.get_market_raw(ticker)
    except Exception as e:
        log.debug("settlement.fetch_failed", ticker=ticker, err=str(e)[:100])
        return False
    result = (market or {}).get("result", "") or ""
    status = (market or {}).get("status", "") or ""
    # Empty result + active status → game still going (or hasn't tipped)
    if result not in ("yes", "no"):
        return False
    side = trade.get("side") or ""
    we_won = (result == side)
    pnl = _settlement_pnl(
        side=side,
        fill_price=float(trade.get("fill_price") or 0),
        contracts=int(trade.get("contracts") or 0),
        our_side_won=we_won,
    )
    journal.update_settlement(
        ticker=ticker, opened_ts=trade["opened_ts"],
        settled_outcome=1 if we_won else 0,
        settlement_pnl_usd=round(pnl, 4),
    )
    log.info("settlement.recorded",
             ticker=ticker, side=side, result=result,
             we_won=we_won, settlement_pnl=round(pnl, 2),
             actual_pnl=trade.get("pnl_usd"))
    return True


async def backfill_loop(
    client: KalshiClient, journal: Journal, *, interval_seconds: int = 600,
) -> None:
    """Run forever, checking pending settlements every `interval_seconds`.

    Default is 10 min — settlements happen ~hourly on game completion,
    so this keeps the journal fresh without hammering Kalshi.
    """
    while True:
        try:
            pending = journal.trades_pending_settlement(limit=500)
            if pending:
                log.info("settlement.batch_start", count=len(pending))
                updated = 0
                # Sequentially to avoid spamming Kalshi; fast enough.
                for trade in pending:
                    ok = await _check_one(client, journal, trade)
                    if ok:
                        updated += 1
                    # 50ms between calls — gentle on the rate limiter
                    await asyncio.sleep(0.05)
                log.info("settlement.batch_done",
                         checked=len(pending), updated=updated)
        except Exception:
            log.exception("settlement.loop_error")
        await asyncio.sleep(interval_seconds)
