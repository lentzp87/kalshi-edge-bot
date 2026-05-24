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


async def _settle_shadows(client: KalshiClient, journal: Journal) -> None:
    """Resolve observe-only shadow-model picks against Kalshi outcomes.

    Many shadow signals share a ticker, so we resolve each distinct
    ticker once, then stamp every pending signal on it. settled_outcome
    = 1 if the picked side won, else 0 — that's what shadow_summary()
    turns into a per-model win rate.
    """
    pending = journal.shadow_signals_pending_settlement(limit=2000)
    if not pending:
        return
    # ticker -> "yes" / "no" / None (None = not yet resolved)
    resolved: dict[str, str | None] = {}
    updated = 0
    for sig in pending:
        ticker = sig.get("ticker") or ""
        if ticker not in resolved:
            try:
                market = await client.get_market_raw(ticker)
                res = (market or {}).get("result", "") or ""
                resolved[ticker] = res if res in ("yes", "no") else None
            except Exception:
                resolved[ticker] = None
            await asyncio.sleep(0.05)
        result = resolved[ticker]
        if result is None:
            continue
        won = 1 if result == (sig.get("side") or "") else 0
        try:
            journal.update_shadow_settlement(
                signal_id=int(sig["id"]), settled_outcome=won,
            )
            updated += 1
        except Exception:
            log.exception("settlement.shadow_update_error",
                           signal_id=sig.get("id"))
    log.info("settlement.shadows.done",
             pending=len(pending), tickers=len(resolved), updated=updated)


async def _run_one_batch(client: KalshiClient, journal: Journal, *, label: str) -> None:
    """One pass over pending trades. Always logs start + end so the bot
    is observably doing work even if there's nothing to update."""
    pending = journal.trades_pending_settlement(limit=500)
    log.info(f"settlement.{label}.start", pending=len(pending))
    updated = 0
    skipped_unresolved = 0
    for trade in pending:
        try:
            ok = await _check_one(client, journal, trade)
            if ok:
                updated += 1
            else:
                skipped_unresolved += 1
        except Exception:
            log.exception("settlement.check_error", ticker=trade.get("ticker"))
        # 50ms between calls — gentle on the rate limiter
        await asyncio.sleep(0.05)
    log.info(f"settlement.{label}.done",
             checked=len(pending), updated=updated,
             still_unresolved=skipped_unresolved)
    # Resolve shadow-model picks in the same pass.
    try:
        await _settle_shadows(client, journal)
    except Exception:
        log.exception("settlement.shadows_error")


async def backfill_loop(
    client: KalshiClient, journal: Journal, *, interval_seconds: int = 600,
) -> None:
    """Run forever, checking pending settlements.

    Pattern:
      1. Immediately on startup: one full pass (loud logging) so the
         dashboard populates within seconds of deploy.
      2. Then every `interval_seconds` (default 10 min): another pass
         to catch newly-resolved games.

    Settlements happen ~hourly as games finish, so 10 min cadence keeps
    the journal fresh without hammering Kalshi.
    """
    # Startup: drain anything that's already resolved from the journal.
    try:
        await _run_one_batch(client, journal, label="startup")
    except Exception:
        log.exception("settlement.startup_error")

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await _run_one_batch(client, journal, label="periodic")
        except Exception:
            log.exception("settlement.loop_error")
