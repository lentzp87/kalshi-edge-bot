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

from dataclasses import replace as _dc_replace

from .config import env_config, file_config
from .cross_exchange import find_spreads
from . import cross_exchange_state
from .decision import evaluate
from .execution import Executor
from . import fast_path
from .journal import Journal
from .kalshi_client import KalshiClient, Market
from .kalshi_ws import KalshiWebSocket
from .models import model_for_category
from .polymarket_client import list_active_markets as poly_list_active_markets
from .risk import RiskEngine
from .scanner import Scanner
from . import shadow_models
from .orphan_recovery import recover_orphans
from .settlement_backfill import backfill_loop as settlement_loop
from .whale_tracker import WhaleTracker

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------- WS cache
# Most-recent Market dataclass per ticker, populated by the scanner loop.
# The WS tick callback consults this cache to reconstruct a fresh Market
# (with updated bid/ask/last_price/volume) without needing to re-fetch
# metadata (title, category, raw fields) over REST. Ticks for tickers we
# haven't scanned yet are no-ops.
_last_seen_markets: dict[str, Market] = {}


def _market_from_tick(cached: Market, ws_msg: dict) -> Market:
    """Reconstruct a Market with fresh prices from a WS ticker_v2 frame.
    Preserves metadata (title, category, raw event_ticker, etc.) from the
    cached scanner snapshot.

    Kalshi ticker_v2 payload shape varies a bit between deployments; we
    try the obvious field names with safe fallbacks.
    """
    body = ws_msg.get("msg") or ws_msg
    def _f(k: str, default: float) -> float:
        v = body.get(k)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default
    new_yes_bid = _f("yes_bid", cached.yes_bid)
    new_yes_ask = _f("yes_ask", cached.yes_ask)
    new_last    = _f("price", _f("last_price", cached.last_price))
    new_volume  = int(_f("volume", cached.volume))
    # Update the raw dict's bid/ask sizes if the tick carries them — the
    # whale tracker reads yes_bid_size_fp / yes_ask_size_fp from raw.
    new_raw = dict(cached.raw)
    for k in ("yes_bid_size_fp", "yes_ask_size_fp",
              "no_bid_dollars", "no_ask_dollars"):
        if k in body:
            new_raw[k] = body[k]
    return _dc_replace(
        cached,
        yes_bid=new_yes_bid,
        yes_ask=new_yes_ask,
        last_price=new_last,
        volume=new_volume,
        raw=new_raw,
    )


def _make_ws_tick_handler(
    whale_tracker: WhaleTracker,
    executor: Executor,
    journal: Journal,
) -> "callable":
    """Build the on_tick callback that the WS subscriber will fire on
    every ticker_v2 frame. Closes over the long-lived objects.

    The handler does two things:
      1. Update whale_tracker with the fresh tick. The tracker's sliding
         baseline picks up real moves even at WS tick rate.
      2. If whale_tracker returned a signal AND we have a cached Market
         for this ticker, schedule a fast-path evaluation as a task. The
         scheduler isolates the trade flow from the WS read loop.
    """
    def _handler(ticker: str, mid: float, ws_msg: dict) -> None:
        cached = _last_seen_markets.get(ticker)
        if cached is None:
            return  # not tradeable yet — scanner hasn't touched it
        try:
            fresh = _market_from_tick(cached, ws_msg)
        except Exception:
            log.exception("ws.tick_market_build_failed", ticker=ticker)
            return
        # Update cache so subsequent ticks compound on the fresh prices
        _last_seen_markets[ticker] = fresh
        signal = whale_tracker.update(fresh)
        if signal is None:
            return
        # Whale detected — fast-path eval out-of-band from the WS read loop.
        try:
            asyncio.get_event_loop().create_task(
                fast_path.evaluate_and_submit(
                    fresh, executor=executor, journal=journal,
                    trigger=signal.reason,
                )
            )
        except RuntimeError:
            log.warning("ws.fast_path_no_loop", ticker=ticker)
    return _handler


async def loop_once(
    scanner: Scanner, executor: Executor, journal: Journal,
    whale_tracker: WhaleTracker,
    ws: KalshiWebSocket | None = None,
) -> None:
    scanned = 0
    by_category: dict[str, int] = {}
    opinions = 0
    signals = 0
    seen_tickers: list[str] = []

    async for market in scanner.stream_tradeable_markets():
        scanned += 1
        by_category[market.category] = by_category.get(market.category, 0) + 1

        # Cache full Market for the WS tick path to reconstruct from later.
        _last_seen_markets[market.ticker] = market
        seen_tickers.append(market.ticker)

        # Feed every market through the whale tracker — it logs whale-
        # shaped deltas itself, and the executor consults it when sizing.
        whale_tracker.update(market)

        # Shadow models — observe-only alternative signal generators
        # (baseline / steam / cross-exchange). They persist picks to the
        # shadow_signals table for per-model backtesting; they do NOT
        # place trades. Wrapped so a shadow failure can't break scanning.
        try:
            shadow_models.run_shadows(market, journal)
        except Exception:
            log.exception("shadow.run_error", ticker=market.ticker)

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

    # Refresh WS subscriptions to cover every tradeable ticker we just
    # saw. subscribe() is idempotent (skips already-subscribed tickers),
    # so we can call it every scan without thrashing. We deliberately
    # don't unsubscribe stale tickers here — tickers that aged out of the
    # tradeable set will just stop producing useful ticks; the dedup +
    # category filters in fast_path drop those evaluations cheaply.
    if ws is not None and seen_tickers:
        try:
            await ws.subscribe(seen_tickers)
        except Exception:
            log.exception("ws.subscribe_failed", count=len(seen_tickers))

    log.info(
        "loop.summary",
        scanned=scanned,
        by_category=by_category,
        opinions=opinions,
        signals=signals,
        ws_subs=len(seen_tickers),
        rejections=scanner.rejection_counts,
    )


async def trading_loop(
    scanner: Scanner, executor: Executor, journal: Journal,
    whale_tracker: WhaleTracker,
    ws: KalshiWebSocket | None = None,
) -> None:
    cfg = file_config()
    while True:
        try:
            await loop_once(scanner, executor, journal, whale_tracker, ws=ws)
        except Exception:
            log.exception("loop.error")
        await asyncio.sleep(cfg.scanner.loop_interval_seconds)


async def journal_cleanup_loop(
    journal: Journal, *,
    interval_seconds: int = 7 * 24 * 3600,
    shadow_max_age_days: int = 14,
) -> None:
    """Periodic prune of journal bloat.

    Every `interval_seconds` (default 7 days):
      - DELETE FROM signals (write-only table, never read).
      - DELETE shadow_signals rows that are unresolved AND older than
        `shadow_max_age_days`. Resolved picks are KEPT — that's the
        backtest.
      - VACUUM to reclaim disk pages.

    Without this the journal grew to 492MB / 1.5M signals + 887k
    shadow_signals on 2026-05-30, blew past the Render starter plan's
    512MB RAM cap, and silently broke dashboard reads. The dedup added
    to log_shadow_signal at the same time prevents the underlying
    growth; this loop is the belt-and-suspenders backstop.

    First run is delayed by `interval_seconds` (not on boot) so a
    crash-restart loop can't thrash VACUUM.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            stats = journal.cleanup_old_rows(
                shadow_max_age_days=shadow_max_age_days,
            )
            log.info("journal.cleanup.done", **stats)
        except Exception:
            log.exception("journal.cleanup.error")


async def slack_digest_loop(
    journal: Journal, *, interval_hours: int = 6,
) -> None:
    """Periodic 6h P&L digest to Slack.

    No-ops gracefully if Slack isn't configured (slack_notifier checks env
    vars on each post and warns once). Interval is configurable via the
    SLACK_DIGEST_INTERVAL_HOURS env var (default 6).
    """
    from .slack_notifier import notify_pnl_digest
    interval_seconds = max(60, int(interval_hours * 3600))
    # Initial delay: wait one full interval before the first ping so the
    # bot doesn't fire an empty digest immediately on every deploy.
    await asyncio.sleep(interval_seconds)
    while True:
        try:
            # Show all four windows at once — periodic digest is more
            # useful as a "state of the bot across timescales" snapshot
            # than a single-window report.
            notify_pnl_digest(journal, windows=[3, 6, 12, 24])
        except Exception:
            log.exception("slack_digest.error")
        await asyncio.sleep(interval_seconds)


async def golf_3ball_loop(*, interval_hours: int = 4) -> None:
    """Periodic golf matchup edge advisor (beta).

    Read-only — pings Slack with +EV golf 3-ball / 2-ball legs from
    DataGolf vs DraftKings. Does NOT place trades. Only re-pings when the
    edge set actually changes, so a quiet slate doesn't spam the channel.
    """
    from .golf_3ball import find_edges, format_slack
    from .slack_notifier import send_text

    last_signature: str | None = None
    await asyncio.sleep(90)  # let the rest of startup settle
    interval_seconds = max(600, int(interval_hours * 3600))
    while True:
        try:
            edges = await find_edges()
            # Signature = the set of (pick, round) pairs — re-ping only
            # when the actual edges change, not every interval.
            signature = "|".join(
                sorted(f"{e.pick}:{e.round_num}:{e.market}" for e in edges)
            )
            if edges and signature != last_signature:
                send_text(format_slack(edges))
                last_signature = signature
                log.info("golf_3ball.pinged", edges=len(edges))
            else:
                log.info("golf_3ball.scan", edges=len(edges),
                         changed=(signature != last_signature))
        except Exception:
            log.exception("golf_3ball.error")
        await asyncio.sleep(interval_seconds)


async def golf_leader_loop(*, interval_minutes: int = 10) -> None:
    """Live golf round-leader alerter (beta).

    Polls every ~10 min during play. Pings Slack when a mid-tier golfer
    is leading / within a stroke with holes to play AND DataGolf's
    in-play probability beats the DraftKings price. Read-only.

    Dedup: each (dg_id, round) is alerted at most once — one heads-up
    per golfer per round, not a re-ping every 10 min.
    """
    from .golf_leader import find_leader_alerts, format_slack
    from .slack_notifier import send_text

    alerted: set[tuple[int, int]] = set()
    await asyncio.sleep(120)  # let startup settle
    interval_seconds = max(120, int(interval_minutes * 60))
    while True:
        try:
            alerts = await find_leader_alerts()
            fresh = [a for a in alerts if (a.dg_id, a.round_num) not in alerted]
            if fresh:
                send_text(format_slack(fresh))
                for a in fresh:
                    alerted.add((a.dg_id, a.round_num))
                log.info("golf_leader.pinged", new=len(fresh), total=len(alerts))
            else:
                log.info("golf_leader.scan", qualifying=len(alerts), new=0)
        except Exception:
            log.exception("golf_leader.error")
        await asyncio.sleep(interval_seconds)


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

    # ---- LIVE PROBE OVERRIDES (2026-06-10) ---------------------------
    # When MODE=live, swap to probe posture WITHOUT touching the paper
    # tuning: scope the scanner to the allowlisted series and tighten
    # the risk caps from the live_* yaml values. Paper mode skips all
    # of this, so the broad paper experiment keeps running unchanged.
    # See the probe block at the bottom of config.yaml.
    if env.mode == "live":
        allow = set(cfg.scanner.live_series_allowlist)
        if allow:
            before = len(cfg.scanner.series_tickers)
            cfg.scanner.series_tickers = [
                s for s in cfg.scanner.series_tickers if s in allow
            ]
            log.info("live.series_allowlist", before=before,
                     after=len(cfg.scanner.series_tickers),
                     series=cfg.scanner.series_tickers)
        r = cfg.risk
        if r.live_max_position_size_usd is not None:
            r.max_position_size_usd = r.live_max_position_size_usd
        if r.live_max_concurrent_positions is not None:
            r.max_concurrent_positions = r.live_max_concurrent_positions
        if r.live_max_daily_loss_usd is not None:
            r.max_daily_loss_usd = r.live_max_daily_loss_usd
        if r.live_whale_max_position_size_usd is not None:
            r.whale_max_position_size_usd = r.live_whale_max_position_size_usd
        if cfg.live_bankroll_usd is not None:
            cfg.bankroll_usd = cfg.live_bankroll_usd
        log.info("live.probe_overrides", bankroll=cfg.bankroll_usd,
                 max_pos=r.max_position_size_usd,
                 max_concurrent=r.max_concurrent_positions,
                 daily_kill=r.max_daily_loss_usd,
                 whale_cap=r.whale_max_position_size_usd)

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

    # Kalshi WebSocket — now wired to the fast-path executor. On every
    # ticker_v2 frame we reconstruct a fresh Market from the scanner's
    # most-recent cached snapshot, feed it into whale_tracker (which uses
    # a ~10s sliding baseline so tick-rate updates produce real deltas),
    # and if a whale signal fires we schedule a fast-path eval out-of-band.
    # Existing executor dedup / cooldown / risk all still gate the trade.
    ws = KalshiWebSocket(client)
    ws.on_tick(_make_ws_tick_handler(whale_tracker, executor, journal))

    try:
        await asyncio.gather(
            trading_loop(scanner, executor, journal, whale_tracker, ws=ws),
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
            # 6h Slack P&L digest. Reads SLACK_DIGEST_INTERVAL_HOURS env
            # var for the interval; defaults to 6.
            slack_digest_loop(
                journal,
                interval_hours=int(
                    _os.environ.get("SLACK_DIGEST_INTERVAL_HOURS", "6")
                ),
            ),
            # Golf 3-ball / 2-ball edge advisor (beta). Read-only — pings
            # Slack with +EV DataGolf-vs-DraftKings matchup legs.
            golf_3ball_loop(
                interval_hours=int(
                    _os.environ.get("GOLF_3BALL_INTERVAL_HOURS", "4")
                ),
            ),
            # Golf round-leader alerter (beta). Read-only — pings Slack
            # when a mid-tier golfer leads/near-leads with holes to play
            # and DataGolf's in-play prob beats the DraftKings price.
            golf_leader_loop(
                interval_minutes=int(
                    _os.environ.get("GOLF_LEADER_INTERVAL_MINUTES", "10")
                ),
            ),
            # WElo seed — one-shot. Downloads recent ATP/WTA match
            # history and builds independent tennis ratings. Until it
            # completes, welo.win_probability() returns None and the
            # tennis model just uses Pinnacle. Observe-only for now.
            _welo_seed_once(),
            # Weekly journal prune. Truncates the write-only signals
            # table and drops stale unresolved shadow picks, then
            # VACUUMs. Without this the DB grew to 492MB and blew past
            # the 512MB RAM cap on 2026-05-30. log_shadow_signal dedup
            # is the primary defense; this is the backstop.
            journal_cleanup_loop(journal),
        )
    finally:
        await client.aclose()


async def _welo_seed_once() -> None:
    """Build WElo ratings once at startup, then exit. Wrapped so a
    seed failure can't take down the asyncio.gather group."""
    try:
        from . import welo
        await welo.seed()
    except Exception:
        log.exception("welo.seed_wrapper_error")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
