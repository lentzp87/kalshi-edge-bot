"""Orphan position recovery on bot startup.

Render redeploys (and any other process restart) kill all in-process
watcher tasks. Open positions in the journal then become orphaned —
no one is updating their current_mid, no one fires their TP/SL, and
they sit forever with `closed_ts IS NULL` even though the underlying
game has long resolved.

This module runs ONCE at bot startup to clean those up:

  1. Pull all open positions from the journal.
  2. For each, fetch Kalshi's current market data.
  3. If Kalshi reports a `result` (game resolved), close the position
     with the actual settlement P&L: $1 if our side won, $0 if it
     lost, minus the entry fee. Exit reason = "orphan_settlement".
  4. If still pending, leave it alone — watcher restoration would be
     needed to re-manage these.

The settlement_pnl column on the trade row remains accurate; this
update writes to the actual pnl_usd / closed_ts / exit_price fields
so the dashboard treats it as a normally-closed trade.
"""

from __future__ import annotations

import structlog

from .fee_model import fee_per_contract_dollars
from .journal import Journal
from .kalshi_client import KalshiClient

log = structlog.get_logger(__name__)


async def recover_orphans(client: KalshiClient, journal: Journal) -> int:
    """Settle resolved-but-still-open positions in the journal.

    Returns the number of positions settled.
    """
    open_positions = journal.open_positions()
    log.info("orphan_recovery.start", open_count=len(open_positions))
    if not open_positions:
        log.info("orphan_recovery.done",
                 closed=0, pending=0, failed=0, note="no open positions")
        return 0

    closed = 0
    pending = 0
    failed = 0

    for pos in open_positions:
        ticker = pos.get("ticker") or ""
        try:
            market = await client.get_market_raw(ticker)
        except Exception as e:
            log.warning("orphan_recovery.fetch_failed",
                        ticker=ticker, err=str(e)[:100])
            failed += 1
            continue

        result = (market or {}).get("result", "") or ""
        status = (market or {}).get("status", "") or ""

        if result not in ("yes", "no"):
            pending += 1
            log.info("orphan_recovery.still_pending",
                     ticker=ticker, status=status, market_result=result,
                     opened_ts=pos.get("opened_ts"),
                     held_hours=_hours_since(pos.get("opened_ts")))
            continue

        side = (pos.get("side") or "").lower()
        we_won = (result == side)
        contracts = int(pos.get("contracts") or 0)
        fill_price = float(pos.get("fill_price") or 0)

        # Per-contract P&L at settlement: $1 - fill if won, -fill if lost.
        per_contract = (1.0 - fill_price) if we_won else (-fill_price)
        gross = per_contract * contracts
        # Only entry fee — Kalshi auto-settles, no exit fee.
        entry_fee = fee_per_contract_dollars(fill_price) * contracts
        pnl_usd = gross - entry_fee
        # Sanity clamp: a long position can never lose more than its size.
        size_usd = float(pos.get("size_usd") or 0)
        if pnl_usd < -size_usd:
            pnl_usd = -size_usd

        # Settlement "exit price" is the binary outcome: 1 if won, 0 if lost.
        exit_price = 1.0 if we_won else 0.0

        journal.log_close(
            ticker=ticker,
            exit_price=exit_price,
            pnl_usd=round(pnl_usd, 4),
            fees_usd=round(entry_fee, 4),
            reason="orphan_settlement",
        )
        closed += 1
        log.info("orphan_recovery.settled",
                 ticker=ticker, side=side, result=result,
                 we_won=we_won, pnl=round(pnl_usd, 2),
                 fill=round(fill_price, 3),
                 held_hours=_hours_since(pos.get("opened_ts")))

    log.info("orphan_recovery.done",
             closed=closed, pending=pending, failed=failed)
    return closed


def _hours_since(iso_ts: str | None) -> float | None:
    """Hours between iso_ts and now. None if input is bad."""
    if not iso_ts:
        return None
    try:
        from datetime import datetime, timezone
        s = iso_ts.replace("Z", "+00:00") if iso_ts.endswith("Z") else iso_ts
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return round((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 1)
    except (ValueError, AttributeError):
        return None
