"""Orchestrator loop.

Runs two coroutines concurrently:
  - the trading loop (scan -> model -> decide -> execute)
  - the FastAPI dashboard (uvicorn embedded)

Single-process design so Render's persistent disk only needs one service.
"""

from __future__ import annotations

import asyncio
import os

import structlog
import uvicorn

from .config import env_config, file_config
from .decision import evaluate
from .execution import Executor
from .journal import Journal
from .kalshi_client import KalshiClient
from .models import model_for_category
from .risk import RiskEngine
from .scanner import Scanner
from .settlement_backfill import backfill_loop as settlement_loop

log = structlog.get_logger(__name__)


async def loop_once(scanner: Scanner, executor: Executor, journal: Journal) -> None:
    scanned = 0
    by_category: dict[str, int] = {}
    opinions = 0
    signals = 0

    async for market in scanner.stream_tradeable_markets():
        scanned += 1
        by_category[market.category] = by_category.get(market.category, 0) + 1

        model = model_for_category(market.category)
        if not model or not getattr(model, "enabled", True):
            continue
        est = await model.estimate(market)
        if not est:
            continue

        opinions += 1
        edge = est.p_yes - market.mid
        journal.log_signal(market, model_p=est.p_yes, edge=edge,
                           confidence=est.confidence, reason=est.reason)

        signal = evaluate(market, est)
        if signal:
            signals += 1
            await executor.submit(signal, market)

    log.info(
        "loop.summary",
        scanned=scanned,
        by_category=by_category,
        opinions=opinions,
        signals=signals,
        rejections=scanner.rejection_counts,
    )


async def trading_loop(scanner: Scanner, executor: Executor, journal: Journal) -> None:
    cfg = file_config()
    while True:
        try:
            await loop_once(scanner, executor, journal)
        except Exception:
            log.exception("loop.error")
        await asyncio.sleep(cfg.scanner.loop_interval_seconds)


async def dashboard_server() -> None:
    """Embed uvicorn so dashboard runs in the same process as the trading loop.

    Render injects $PORT for web services; we honor it. For local runs,
    DASHBOARD_PORT from .env wins.
    """
    env = env_config()
    port = int(os.environ.get("PORT", env.dashboard_port))
    config = uvicorn.Config(
        "src.dashboard:app",
        host=env.dashboard_host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    log.info("dashboard.start", port=port)
    await server.serve()


async def amain() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )
    cfg = file_config()
    env = env_config()
    log.info("bot.start", mode=env.mode, kalshi_env=env.kalshi_env, bankroll=cfg.bankroll_usd)

    # Observability: where is the journal writing? Which odds key is in use?
    import os as _os
    from pathlib import Path as _Path
    data_dir_resolved = _Path(env.data_dir).resolve()
    log.info("bot.data_dir",
             value=env.data_dir, resolved=str(data_dir_resolved),
             exists=data_dir_resolved.exists())
    odds_key_env = _os.environ.get("ODDS_API_KEY")
    if odds_key_env:
        masked = odds_key_env[:4] + "..." + odds_key_env[-4:] if len(odds_key_env) >= 8 else "***"
        log.info("bot.odds_api_key", source="env", masked=masked)
    else:
        log.warning("bot.odds_api_key", source="baked",
                    note="ODDS_API_KEY env var not set — using baked-in key (likely dead)")

    client = KalshiClient()
    # Pull Kalshi's full series catalog ONCE at startup so every market
    # we see can be categorized by Kalshi's own taxonomy ("Sports",
    # "Climate and Weather", etc.) — replaces the brittle ticker-prefix guesses.
    await client.load_series_categories()

    # Dynamic discovery: build the list of series_tickers to scan.
    discovered: list[str] = []
    if cfg.models.sports.enabled:
        from .models.sports import SERIES_REGISTRY as SPORTS_SERIES
        from .models.tennis import TENNIS_SERIES
        from .models.ufc import UFC_SERIES
        from .models.soccer import SOCCER_SERIES
        from .models.golf import GOLF_SERIES
        sports = list(SPORTS_SERIES.keys())
        tennis = sorted(TENNIS_SERIES)
        ufc = sorted(UFC_SERIES)
        soccer = sorted(SOCCER_SERIES)
        golf = sorted(GOLF_SERIES)
        log.info("scanner.dynamic_series.sports", count=len(sports), series=sports)
        log.info("scanner.dynamic_series.tennis", count=len(tennis), series=tennis)
        log.info("scanner.dynamic_series.ufc",    count=len(ufc),    series=ufc)
        log.info("scanner.dynamic_series.soccer", count=len(soccer), series=soccer)
        log.info("scanner.dynamic_series.golf",   count=len(golf),   series=golf)
        discovered += sports + tennis + ufc + soccer + golf

    if discovered:
        cfg.scanner.series_tickers = discovered

    journal = Journal()
    risk = RiskEngine()
    scanner = Scanner(client)
    executor = Executor(client, risk, journal)

    try:
        await asyncio.gather(
            trading_loop(scanner, executor, journal),
            dashboard_server(),
            # Periodic settlement backfill: for each closed trade, look up
            # the Kalshi market's resolution and compute "what if held to
            # settlement" P&L. Surfaces whether our exits are leaving
            # money on the table.
            settlement_loop(client, journal, interval_seconds=600),
        )
    finally:
        await client.aclose()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
