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
from .cross_exchange import find_spreads
from . import cross_exchange_state
from .decision import evaluate
from .execution import Executor
from .journal import Journal
from .kalshi_client import KalshiClient
from .kalshi_ws import KalshiWebSocket
from .models import model_for_category
from .polymarket_client import list_active_markets as poly_list_active_markets
from .risk import RiskEngine
from .scanner import Scanner
from .orphan_recovery import recover_orphans
from .settlement_backfill import backfill_loop as settlement_loop
from .whale_tracker import WhaleTracker

log = structlog.get_logger(__name__)


async def loop_once(
    scanner: Scanner, executor: Executor, journal: Journal,
    whale_tracker: WhaleTracker,
) -> None:
    scanned = 0
    by_category: dict[str, int] = {}
    opinions = 0
    signals = 0

    async for market in scanner.stream_tradeable_markets():
        scanned += 1
        by_category[market.category] = by_category.get(market.category, 0) + 1

        # Feed every market through the whale tracker — it logs whale-
        # shaped deltas itself, and the executor consults it when sizing.
        whale_tracker.update(market)

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


async def trading_loop(
    scanner: Scanner, executor: Executor, journal: Journal,
    whale_tracker: WhaleTracker,
) -> None:
    cfg = file_config()
    while True:
        try:
            await loop_once(scanner, executor, journal, whale_tracker)
        except Exception:
            log.exception("loop.error")
        await asyncio.sleep(cfg.scanner.loop_interval_seconds)


async def cross_exchange_loop(
    client: KalshiClient, *, interval_seconds: int = 300,
    max_pages: int = 5,
) -> None:
    """Periodic Kalshi-vs-Polymarket spread snapshot.

    Read-only and out-of-band: doesn't touch the trading loop. Stores its
    output in `cross_exchange_state` for the dashboard to surface.

    `max_pages` caps the number of Kalshi paginated requests per cycle so
    we don't hammer the API; with limit=200 that's ~1000 markets, which is
    more than enough for the cross-exchange comparison since most Kalshi
    markets won't have a Polymarket twin anyway.
    """
    while True:
        try:
            # Fetch Kalshi: paginate "open" markets up to max_pages.
            kalshi_markets = []
            cursor: str | None = None
            for _ in range(max_pages):
                batch, cursor = await client.list_markets(
                    status="open", limit=200, cursor=cursor,
                )
                kalshi_markets.extend(batch)
                if not cursor:
                    break

            poly_markets = await poly_list_active_markets()

            spreads = find_spreads(
                kalshi_markets=kalshi_markets,
                polymarket_markets=poly_markets,
            )
            cross_exchange_state.update(
                n_kalshi=len(kalshi_markets),
                n_polymarket=len(poly_markets),
                spreads=spreads,
            )
            log.info(
                "cross_exchange.snapshot",
                n_kalshi=len(kalshi_markets),
                n_polymarket=len(poly_markets),
                spreads=len(spreads),
            )
        except Exception:
            log.exception("cross_exchange.error")
        await asyncio.sleep(interval_seconds)


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

    # Slack: presence check for trade-notification credentials. Fail-silent
    # at send time; this just surfaces the config state in deploy logs.
    slack_hook = _os.environ.get("SLACK_WEBHOOK_URL")
    slack_token = _os.environ.get("SLACK_BOT_TOKEN")
    slack_channel = _os.environ.get("SLACK_CHANNEL")
    if slack_hook:
        log.info("bot.slack", transport="webhook", configured=True)
    elif slack_token and slack_channel:
        log.info("bot.slack", transport="bot_token",
                 channel=slack_channel, configured=True)
    else:
        log.info("bot.slack", configured=False,
                 note="set SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN+SLACK_CHANNEL")

    # DataGolf is the Tier-1 truth source for the golf model. Log presence at
    # startup so deploy logs show whether we're running with the sharper
    # provider or falling back to The Odds API outright odds.
    dg_key_env = _os.environ.get("DATAGOLF_API_KEY")
    if dg_key_env:
        dg_masked = (
            dg_key_env[:4] + "..." + dg_key_env[-4:]
            if len(dg_key_env) >= 8 else "***"
        )
        log.info("bot.datagolf_key", source="env", masked=dg_masked)
    else:
        log.info(
            "bot.datagolf_key", source="missing",
            note="DATAGOLF_API_KEY not set; golf falls back to Odds API outrights",
        )

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
        from .models.cricket import CRICKET_SERIES
        sports = list(SPORTS_SERIES.keys())
        tennis = sorted(TENNIS_SERIES)
        ufc = sorted(UFC_SERIES)
        soccer = sorted(SOCCER_SERIES)
        golf = sorted(GOLF_SERIES)
        cricket = sorted(CRICKET_SERIES)
        log.info("scanner.dynamic_series.sports", count=len(sports), series=sports)
        log.info("scanner.dynamic_series.tennis", count=len(tennis), series=tennis)
        log.info("scanner.dynamic_series.ufc",    count=len(ufc),    series=ufc)
        log.info("scanner.dynamic_series.soccer", count=len(soccer), series=soccer)
        log.info("scanner.dynamic_series.golf",   count=len(golf),   series=golf)
        log.info("scanner.dynamic_series.cricket", count=len(cricket), series=cricket)
        discovered += sports + tennis + ufc + soccer + golf + cricket

    if discovered:
        cfg.scanner.series_tickers = discovered

    journal = Journal()
    risk = RiskEngine()
    scanner = Scanner(client)
    whale_tracker = WhaleTracker()
    executor = Executor(client, risk, journal, whale_tracker=whale_tracker)

    # Orphan recovery — runs ONCE before the trading loop starts.
    # Render redeploys kill in-process watchers, leaving open positions
    # in the journal with no one to close them. This walks the open list,
    # checks Kalshi for resolution, and settles any whose game has ended.
    try:
        await recover_orphans(client, journal)
    except Exception:
        log.exception("orphan_recovery.error")

    # Kalshi WebSocket — read-only ticker subscriber. No-ops gracefully if
    # the `websockets` library isn't installed (REST polling continues
    # regardless). Currently just logs ticks; integration with the in-game
    # model is a follow-up.
    ws = KalshiWebSocket(client)

    try:
        await asyncio.gather(
            trading_loop(scanner, executor, journal, whale_tracker),
            dashboard_server(),
            # Periodic settlement backfill: for each closed trade, look up
            # the Kalshi market's resolution and compute "what if held to
            # settlement" P&L. Surfaces whether our exits are leaving
            # money on the table.
            settlement_loop(client, journal, interval_seconds=600),
            # WebSocket: read-only smoke test of the Kalshi WS pipe.
            ws.run(),
            # Polymarket cross-exchange spread snapshot every 5 min.
            cross_exchange_loop(client, interval_seconds=300),
        )
    finally:
        await client.aclose()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
