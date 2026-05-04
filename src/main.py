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

    client = KalshiClient()
    journal = Journal()
    risk = RiskEngine()
    scanner = Scanner(client)
    executor = Executor(client, risk, journal)

    try:
        await asyncio.gather(
            trading_loop(scanner, executor, journal),
            dashboard_server(),
        )
    finally:
        await client.aclose()


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
