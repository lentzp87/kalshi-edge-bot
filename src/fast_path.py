"""Fast-path execution: triggered by a WebSocket-derived whale signal,
runs the model → decide → submit pipeline OUT-OF-BAND from the scanner.

Latency budget vs scanner-driven path:
  scanner: ~30s poll interval + 2-5s scan = 30-35s whale-to-trade
  fast_path: ~1-2s WS tick latency + <100ms model eval = 1-2s whale-to-trade

Safety:
  * All existing executor checks still gate the trade (per-event dedup,
    cooldown_until, risk.approve, position-cap, size minimums). We do
    not bypass any of them. We just invoke executor.submit() from a
    different callsite.
  * Per-ticker cooldown inside this module prevents us from spamming
    the pipeline on a price-jump followed by a 1s aftershock — we'll
    only fast-path a given ticker at most once every 5 seconds.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from .decision import evaluate
from .kalshi_client import Market
from .models import model_for_category

log = structlog.get_logger(__name__)


# Per-ticker last-fast-path-time so we don't loop on aftershocks.
_last_fp: dict[str, float] = {}
_FP_COOLDOWN_SECONDS = 5.0


async def evaluate_and_submit(
    market: Market,
    *,
    executor: Any,
    journal: Any,
    trigger: str,
) -> None:
    """Run model → decide → submit for `market` outside the scanner loop.

    Called when a WS-derived whale signal fires. The executor's existing
    dedup / cooldown / risk checks are the source of truth for whether
    the trade actually goes through — we just shortcut the path from
    "tick arrived" to "submit called".
    """
    ticker = market.ticker or ""
    if not ticker:
        return

    # Per-ticker cooldown — guard against tick storms.
    now = time.time()
    if (now - _last_fp.get(ticker, 0)) < _FP_COOLDOWN_SECONDS:
        return
    _last_fp[ticker] = now

    try:
        model = model_for_category(market.category)
        if not model or not getattr(model, "enabled", True):
            return
        est = await model.estimate(market)
        if not est:
            return

        edge = est.p_yes - market.mid
        # Same journaling shape as the scanner loop — every opinion gets
        # logged whether or not it produces a signal.
        journal.log_signal(
            market, model_p=est.p_yes, edge=edge,
            confidence=est.confidence, reason=est.reason,
        )

        signal = evaluate(market, est)
        if not signal:
            return

        log.info(
            "fast_path.firing",
            ticker=ticker, side=signal.side,
            edge_bp=round(edge * 100, 1),
            price=signal.price_cents / 100,
            trigger=trigger,
        )
        await executor.submit(signal, market)
    except Exception:
        log.exception("fast_path.error", ticker=ticker)
