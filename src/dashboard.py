"""FastAPI dashboard with sports CLV breakdowns.

Endpoints:
  GET /health              liveness (used by Render health check)
  GET /positions           currently open
  GET /pnl                 daily P&L for last 30 days
  GET /trades?limit=100    recent trade log
  GET /edge                realized vs predicted edge buckets
  GET /stats               aggregate metrics + per-dimension breakdowns
  GET /                    full dashboard UI (auto-refreshing)
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import env_config, file_config
from . import cross_exchange_state
from .exit_simulator import aggregate_exit_policies, edge_bucket
from .journal import Journal

app = FastAPI(title="Kalshi Edge Bot")
_journal = Journal()


# ----- Helpers --------------------------------------------------------------

def _is_clean_trade(t: dict) -> bool:
    """Filter out trades with corrupted P&L from the early-deploy bug era."""
    pnl = t.get("pnl_usd")
    size = t.get("size_usd") or 0
    if pnl is None:
        return False
    return abs(pnl) <= max(size * 3.0, 200.0)


def _sport_from_ticker(ticker: str) -> str:
    """Pull a friendly sport label from the Kalshi market ticker."""
    if not ticker:
        return "?"
    series = ticker.split("-", 1)[0].upper()
    mapping = {
        "KXNBAGAME": "NBA",
        "KXNFLGAME": "NFL",
        "KXMLBGAME": "MLB",
        "KXNHLGAME": "NHL",
    }
    return mapping.get(series, series)


_RE_PROVIDER = re.compile(r"book\[([^\]]+)\]")
_RE_BOOKS = re.compile(r"\((\d+)\s+books?\)")
_RE_OUR_SIDE = re.compile(r"our=\w+\((home|away)\)")
_RE_TIP_MIN = re.compile(r"tip in ([\d.]+)min")
_RE_NET_EDGE = re.compile(r"net_edge=([+-]?[\d.]+)")
_RE_GROSS = re.compile(r"gross ([+-]?[\d.]+)")
_RE_P_YES = re.compile(r"p_yes=([\d.]+)")
# Whale-boost fields appended by execution.py when the whale tracker
# aligned with our side. `class=aggressive|burst|resting|other` and
# `magnitude=<bucket>` (e.g. "price_7-10c", "burst_25-50k").
_RE_WHALE_CLASS = re.compile(r"WHALE_ALIGNED class=(\w+)")
_RE_WHALE_MAGNITUDE = re.compile(r"magnitude=(\S+)")


def _parse_reason(reason: str | None) -> dict:
    """Extract structured signal-time fields from the journal reason string."""
    out: dict = {}
    if not reason:
        return out
    if m := _RE_PROVIDER.search(reason):
        provider_full = m.group(1)
        # "odds_api (8 books)" or "espn:Draft Kings"
        if provider_full.startswith("odds_api"):
            out["provider"] = "odds_api"
            if bm := _RE_BOOKS.search(provider_full):
                out["books"] = int(bm.group(1))
        elif provider_full.startswith("espn"):
            out["provider"] = "espn"
        else:
            out["provider"] = provider_full
    if m := _RE_OUR_SIDE.search(reason):
        out["our_side"] = m.group(1)  # "home" or "away"
    if m := _RE_TIP_MIN.search(reason):
        try:
            out["mins_to_tip"] = float(m.group(1))
        except ValueError:
            pass
    if m := _RE_NET_EDGE.search(reason):
        try:
            out["net_edge"] = float(m.group(1))
        except ValueError:
            pass
    if m := _RE_GROSS.search(reason):
        try:
            out["gross_edge"] = float(m.group(1))
        except ValueError:
            pass
    if m := _RE_P_YES.search(reason):
        try:
            out["p_yes"] = float(m.group(1))
        except ValueError:
            pass
    if m := _RE_WHALE_CLASS.search(reason):
        out["whale_class"] = m.group(1)
    if m := _RE_WHALE_MAGNITUDE.search(reason):
        out["whale_magnitude"] = m.group(1)
    return out


def _confidence_bucket(p_yes: float | None, side: str | None) -> str | None:
    """Bucket the model's probability for OUR side.

    `p_yes` from the reason string is always the YES side's prob. If we
    bet NO, our side's prob = 1 - p_yes. Buckets target the user's
    "65% to win" sweet spot.
    """
    if p_yes is None or side not in ("yes", "no"):
        return None
    our_p = p_yes if side == "yes" else (1 - p_yes)
    if our_p < 0.50: return "<50%"
    if our_p < 0.55: return "50-55%"
    if our_p < 0.60: return "55-60%"
    if our_p < 0.65: return "60-65%"
    if our_p < 0.70: return "65-70%"
    if our_p < 0.80: return "70-80%"
    return "80%+"


def _entry_price_bucket(fill_price: float | None) -> str:
    if fill_price is None:
        return "?"
    p = fill_price
    if p < 0.55: return "<55"
    if p < 0.65: return "55-65"
    if p < 0.75: return "65-75"
    if p < 0.85: return "75-85"
    return "85+"


def _side_entry_bucket(side: str | None, fill_price: float | None) -> str | None:
    """Compose `side × entry-bucket` key for the asymmetric-floor analysis
    panel. Returns e.g. 'yes <55', 'no 65-75'. None drops the row.

    Smaller buckets than `_entry_price_bucket` to surface differences
    in the cheap end where the new 0.35 floor matters most.
    """
    if side not in ("yes", "no") or fill_price is None:
        return None
    p = float(fill_price)
    if p < 0.40:    bucket = "<40"
    elif p < 0.50:  bucket = "40-50"
    elif p < 0.60:  bucket = "50-60"
    elif p < 0.70:  bucket = "60-70"
    elif p < 0.80:  bucket = "70-80"
    else:           bucket = "80+"
    return f"{side} {bucket}"


def _tip_bucket(mins: float | None) -> str:
    if mins is None:
        return "?"
    if mins < 30: return "0-30m"
    if mins < 60: return "30-60m"
    if mins < 120: return "1-2h"
    if mins < 240: return "2-4h"
    return "4h+"


def _hold_bucket(opened_ts: str | None, closed_ts: str | None) -> str | None:
    """Bucket the trade's hold duration. Mirrors what we'd want to see
    if short-hold trades cluster around stop_losses on noise."""
    if not opened_ts or not closed_ts:
        return None
    try:
        from datetime import datetime as _dt
        o = _dt.fromisoformat(opened_ts.replace("Z", "+00:00"))
        c = _dt.fromisoformat(closed_ts.replace("Z", "+00:00"))
        mins = (c - o).total_seconds() / 60
    except (ValueError, AttributeError):
        return None
    if mins < 2:    return "<2m"
    if mins < 5:    return "2-5m"
    if mins < 15:   return "5-15m"
    if mins < 60:   return "15-60m"
    if mins < 240:  return "1-4h"
    return "4h+"


def _open_unrealized_snapshot(open_positions: list[dict]) -> dict:
    """Mark-to-market across all open positions. Unrealized P&L doesn't
    depend on a window — it's just a snapshot of where the book stands
    right now. Positions without a `current_mid` are counted in `n_total`
    but excluded from the sum (`n_marked` tracks how many were summable).
    """
    from .fee_model import fee_per_contract_dollars
    total = 0.0
    n_marked = 0
    for p in open_positions:
        mid = p.get("current_mid")
        fill = p.get("fill_price")
        contracts = p.get("contracts")
        if mid is None or fill is None or not contracts:
            continue
        try:
            mid_f = float(mid)
            fill_f = float(fill)
            c = int(contracts)
        except (TypeError, ValueError):
            continue
        # Estimated exit fee at the current mid. Entry fee already sunk
        # in the fill price, so don't double-count it.
        exit_fee = fee_per_contract_dollars(mid_f) * c
        total += (mid_f - fill_f) * c - exit_fee
        n_marked += 1
    return {
        "unrealized_usd": round(total, 2),
        "n_open_total": len(open_positions),
        "n_open_marked": n_marked,
    }


def _windowed_pnl(closed_trades: list[dict], window_hours: int) -> dict:
    """Realized P&L for trades closed within the last `window_hours`.

    Expects `closed_trades` to be the enriched, cleaned, chronological
    list already filtered to clean (non-corrupted) trades. The unrealized
    half is computed separately by _open_unrealized_snapshot since it's
    window-independent.
    """
    from datetime import datetime, timezone, timedelta
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    rows = [
        t for t in closed_trades
        if t.get("closed_ts") and t["closed_ts"] >= cutoff_iso
    ]
    if not rows:
        return {
            "window_hours": window_hours, "n": 0, "wins": 0, "losses": 0,
            "realized_usd": 0.0, "win_rate": 0.0,
        }
    realized = sum(float(t.get("pnl_usd") or 0) for t in rows)
    wins = sum(1 for t in rows if (t.get("pnl_usd") or 0) > 0)
    losses = sum(1 for t in rows if (t.get("pnl_usd") or 0) < 0)
    n = len(rows)
    return {
        "window_hours": window_hours,
        "n": n,
        "wins": wins,
        "losses": losses,
        "realized_usd": round(realized, 2),
        "win_rate": round(wins / n, 3) if n else 0.0,
    }


def _tennis_summary(closed_trades: list[dict]) -> dict:
    """Combined ATP + WTA P&L tracker. Tennis is the bot's one proven
    surface, so we isolate it: lifetime + 24h + 72h, split by tour.
    """
    from datetime import datetime, timezone, timedelta
    tennis = [t for t in closed_trades
              if (t.get("ticker") or "").split("-", 1)[0]
              in ("KXATPMATCH", "KXWTAMATCH")]

    def _agg(rows: list[dict]) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0, "wins": 0, "losses": 0, "pnl": 0.0, "win_rate": 0.0}
        pnl = sum(float(r.get("pnl_usd") or 0) for r in rows)
        w = sum(1 for r in rows if (r.get("pnl_usd") or 0) > 0)
        return {
            "n": n, "wins": w, "losses": n - w,
            "pnl": round(pnl, 2), "win_rate": round(w / n, 3),
        }

    now = datetime.now(timezone.utc)
    iso24 = (now - timedelta(hours=24)).isoformat()
    iso72 = (now - timedelta(hours=72)).isoformat()
    return {
        "lifetime": _agg(tennis),
        "wta": _agg([t for t in tennis
                     if (t.get("ticker") or "").startswith("KXWTAMATCH")]),
        "atp": _agg([t for t in tennis
                     if (t.get("ticker") or "").startswith("KXATPMATCH")]),
        "last_24h": _agg([t for t in tennis if t.get("closed_ts", "") >= iso24]),
        "last_72h": _agg([t for t in tennis if t.get("closed_ts", "") >= iso72]),
    }


def _bucket_aggregate(rows: list[dict], key_fn) -> list[dict]:
    """Group rows by key_fn(row) and return [{key, n, wins, pnl, avg_pnl, win_rate}]."""
    bucket: dict = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        b = bucket[k]
        b["n"] += 1
        b["pnl"] += r["pnl_usd"]
        if r["pnl_usd"] > 0:
            b["wins"] += 1
    out = []
    for k, v in bucket.items():
        n = v["n"]
        out.append({
            "key": str(k),
            "n": n,
            "wins": v["wins"],
            "pnl": round(v["pnl"], 2),
            "avg_pnl": round(v["pnl"] / n, 2) if n else 0.0,
            "win_rate": round(v["wins"] / n, 3) if n else 0.0,
        })
    return sorted(out, key=lambda x: -abs(x["pnl"]))


# ----- Endpoints ------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/positions")
def positions() -> list[dict]:
    return _journal.open_positions()


@app.get("/trades")
def trades(limit: int = 100) -> list[dict]:
    return _journal.recent_trades(limit=limit)


@app.get("/pnl")
def pnl() -> dict:
    return _journal.daily_pnl()


@app.get("/edge")
def edge() -> dict:
    return _journal.realized_edge_summary()


@app.get("/pnl_digest")
def pnl_digest(
    windows: str = "3,6,12,24",
    window_hours: int | None = None,
    send: bool = False,
) -> dict:
    """Return the same P&L digest the Slack loop sends.

    ?send=true also fires the Slack ping (useful for testing the webhook
    without waiting for the periodic timer).

    ?windows=3,6,12,24 controls which windows show in the table
    (comma-separated hours). Default is the same set the periodic loop
    posts to Slack.

    ?window_hours=N (legacy) forces a single-window digest.
    """
    from .slack_notifier import build_pnl_digest, notify_pnl_digest
    if window_hours is not None:
        text = build_pnl_digest(_journal, window_hours=int(window_hours))
        meta = {"window_hours": int(window_hours)}
        if send:
            notify_pnl_digest(_journal, window_hours=int(window_hours))
    else:
        try:
            win_list = [int(w.strip()) for w in windows.split(",") if w.strip()]
        except ValueError:
            win_list = [3, 6, 12, 24]
        if not win_list:
            win_list = [3, 6, 12, 24]
        text = build_pnl_digest(_journal, windows=win_list)
        meta = {"windows": win_list}
        if send:
            notify_pnl_digest(_journal, windows=win_list)
    return {**meta, "text": text, "sent": bool(send)}


@app.get("/golf_3ball")
async def golf_3ball(send: bool = False, min_edge_pp: float = 0.04) -> dict:
    """Golf matchup edge advisor (beta). Read-only — surfaces +EV golf
    3-ball / 2-ball legs from DataGolf vs DraftKings.

    ?send=true also fires the Slack ping. ?min_edge_pp=N adjusts the
    flag threshold (default 0.04 = 4 percentage points).
    """
    from .golf_3ball import find_edges, format_slack
    edges = await find_edges(min_edge_pp=min_edge_pp)
    text = format_slack(edges)
    if send:
        from .slack_notifier import send_text
        send_text(text)
    return {
        "n_edges": len(edges),
        "min_edge_pp": min_edge_pp,
        "sent": bool(send),
        "text": text,
        "edges": [vars(e) for e in edges],
    }


@app.get("/welo")
def welo(a: str = "", b: str = "", surface: str = "") -> dict:
    """WElo (Weighted Elo) tennis model inspector. Read-only.

    No args: returns seed status. With ?a=Player&b=Player (&surface=clay)
    returns WElo's win probability for that matchup — the bot's
    independent second opinion vs the Pinnacle line.
    """
    from .welo import engine
    status = {
        "seeded": engine.seeded,
        "n_matches": engine.n_matches,
        "n_players": engine.player_count(),
    }
    if a and b:
        p = engine.win_probability(a, b, surface or None)
        status["query"] = {
            "player_a": a, "player_b": b,
            "surface": surface or "overall",
            "p_a_wins": round(p, 4) if p is not None else None,
            "note": None if p is not None else (
                "engine not seeded yet" if not engine.seeded
                else "no rating history for one or both players"),
        }
    return status


@app.get("/golf_leader")
async def golf_leader(send: bool = False, lead_gap: int = 1,
                      min_thru: int = 9, min_edge_pp: float = 0.05) -> dict:
    """Live golf round-leader alerter (beta). Read-only — surfaces
    mid-tier golfers near the lead with holes to play whose DataGolf
    in-play probability beats the DraftKings price.

    ?send=true also fires the Slack ping. lead_gap / min_thru /
    min_edge_pp tune the trigger.
    """
    from .golf_leader import find_leader_alerts, format_slack
    alerts = await find_leader_alerts(
        lead_gap=lead_gap, min_thru=min_thru, min_edge_pp=min_edge_pp,
    )
    text = format_slack(alerts)
    if send:
        from .slack_notifier import send_text
        send_text(text)
    return {
        "n_alerts": len(alerts),
        "sent": bool(send),
        "text": text,
        "alerts": [
            {
                "player": a.player_name, "pos": a.current_pos,
                "score": a.current_score, "strokes_back": a.strokes_back,
                "thru": a.thru, "round": a.round_num,
                "best_market": a.best_market,
                "best_edge_pp": round(a.best_edge_pp, 4),
                "market_edges": a.market_edges,
            }
            for a in alerts
        ],
    }


@app.get("/cross_exchange")
def cross_exchange() -> dict:
    """Latest Kalshi-vs-Polymarket spread snapshot.

    Populated by `cross_exchange_loop` in main.py every ~5 min. Cold
    start returns ts=None and an empty spreads list.
    """
    return cross_exchange_state.latest()


@app.get("/stats")
def stats() -> dict:
    raw = _journal.recent_trades(limit=10000)
    closed = [t for t in raw if t.get("closed_ts")]
    clean = [t for t in closed if _is_clean_trade(t)]

    # Pre-compute parsed signal fields once per trade
    enriched: list[dict] = []
    for t in clean:
        meta = _parse_reason(t.get("reason"))
        enriched.append({
            **t,
            "_sport": _sport_from_ticker(t.get("ticker", "")),
            "_provider": meta.get("provider"),
            "_books": meta.get("books"),
            "_our_side": meta.get("our_side"),
            "_mins_to_tip": meta.get("mins_to_tip"),
            "_net_edge": meta.get("net_edge"),
            "_gross_edge": meta.get("gross_edge"),
            "_p_yes": meta.get("p_yes"),
            "_entry_bucket": _entry_price_bucket(t.get("fill_price")),
            "_tip_bucket": _tip_bucket(meta.get("mins_to_tip")),
            "_confidence_bucket": _confidence_bucket(
                meta.get("p_yes"), t.get("side")),
            "_fav_dog": (
                "favorite" if (t.get("fill_price") or 0) >= 0.5 else "underdog"
            ),
            # None for non-whale-boosted trades — _bucket_aggregate drops Nones.
            "_whale_class": meta.get("whale_class"),
            "_whale_magnitude": meta.get("whale_magnitude"),
        })

    if not enriched:
        return {
            "total_pnl": 0.0, "n_trades": 0, "n_wins": 0, "n_losses": 0,
            "win_rate": 0.0, "avg_pnl": 0.0, "avg_edge_bp": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "open_positions": len(_journal.open_positions()),
            "filtered_out": len(closed),
            "pnl_curve": [], "edge_calibration": [],
            "by_sport": [], "by_provider": [], "by_our_side": [],
            "by_fav_dog": [], "by_entry_bucket": [], "by_tip_bucket": [],
            "by_exit_reason": [], "by_series": [], "by_hold": [],
            "by_confidence": [],
            "by_exit_policy": [], "by_edge_bucket": [],
            "by_whale_class": [], "by_whale_magnitude": [], "by_whale_side": [],
            "by_side_x_entry": [],
            "windowed_pnl": [], "open_mtm": _open_unrealized_snapshot(_journal.open_positions()),
            "tennis_summary": _tennis_summary([]),
            "n_clv": 0, "avg_clv_bp": 0.0, "pct_positive_clv": 0.0,
            "by_sport_clv": [],
            "settlement": {
                "n_resolved": 0, "actual_total": 0.0, "settlement_total": 0.0,
                "delta": 0.0, "better_held_pct": 0.0,
                "stop_loss_n": 0, "stop_loss_actual": 0.0, "stop_loss_settlement": 0.0,
            },
        }

    chronological = sorted(enriched, key=lambda t: t["closed_ts"])
    total = sum(t["pnl_usd"] for t in chronological)
    wins = [t for t in chronological if t["pnl_usd"] > 0]
    losses = [t for t in chronological if t["pnl_usd"] <= 0]
    n = len(chronological)
    avg_edge_pp = sum((t.get("edge") or 0) for t in chronological) / n

    # Cumulative curve
    curve, cum = [], 0.0
    for t in chronological:
        cum += t["pnl_usd"]
        curve.append({
            "ts": t["closed_ts"],
            "cum": round(cum, 2),
            "pnl": round(t["pnl_usd"], 2),
            "ticker": t.get("ticker", ""),
        })

    # Edge calibration (5pp buckets of predicted edge)
    bucket_data: dict = defaultdict(lambda: {"n": 0, "predicted": 0.0, "realized": 0.0, "wins": 0})
    for t in chronological:
        e = (t.get("edge") or 0)
        size = t.get("size_usd") or 0
        b = int(round(e * 100 / 5)) * 5
        bd = bucket_data[b]
        bd["n"] += 1
        bd["predicted"] += e * size
        bd["realized"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            bd["wins"] += 1
    edge_calibration = [
        {
            "bucket_pp": k, "n": v["n"],
            "predicted": round(v["predicted"] / v["n"], 2),
            "realized": round(v["realized"] / v["n"], 2),
            "win_rate": round(v["wins"] / v["n"], 3),
        }
        for k, v in sorted(bucket_data.items())
    ]

    return {
        "total_pnl": round(total, 2),
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / n, 3),
        "avg_pnl": round(total / n, 2),
        "avg_edge_bp": round(avg_edge_pp * 100, 1),
        "best_trade": round(max(t["pnl_usd"] for t in chronological), 2),
        "worst_trade": round(min(t["pnl_usd"] for t in chronological), 2),
        "open_positions": len(_journal.open_positions()),
        "filtered_out": len(closed) - len(clean),
        "pnl_curve": curve,
        "edge_calibration": edge_calibration,
        # Per-dimension breakdowns
        "by_sport": _bucket_aggregate(chronological, lambda r: r["_sport"]),
        "by_provider": _bucket_aggregate(chronological, lambda r: r["_provider"]),
        "by_our_side": _bucket_aggregate(chronological, lambda r: r["_our_side"]),
        "by_fav_dog": _bucket_aggregate(chronological, lambda r: r["_fav_dog"]),
        "by_entry_bucket": _bucket_aggregate(chronological, lambda r: r["_entry_bucket"]),
        "by_tip_bucket": _bucket_aggregate(chronological, lambda r: r["_tip_bucket"]),
        "by_exit_reason": _bucket_aggregate(chronological, lambda r: r.get("exit_reason")),
        "by_series": _bucket_aggregate(
            chronological,
            lambda r: (r.get("ticker") or "").split("-", 1)[0] or "?"
        ),
        # Hold-duration buckets answer "did we exit too early?". If
        # <2m and 2-5m buckets dominate stop_losses with negative P&L,
        # we're being whipped out on noise and should widen SL or
        # delay it (e.g. no SL in first 30 min after entry).
        "by_hold": _bucket_aggregate(
            chronological,
            lambda r: _hold_bucket(r.get("opened_ts"), r.get("closed_ts"))
        ),
        # Model-confidence buckets: how does the model's claimed p_yes
        # for our side correlate with actual win rate? Calibration check.
        "by_confidence": _bucket_aggregate(
            chronological,
            lambda r: r.get("_confidence_bucket")
        ),
        # A/B exit-policy cohort sim: per-policy P&L if every trade had
        # used a different exit rule (TP-only, SL-only, time-only, etc.).
        "by_exit_policy": aggregate_exit_policies(chronological),
        # Edge-magnitude buckets: does bigger predicted edge actually
        # produce bigger realized P&L?
        "by_edge_bucket": _bucket_aggregate(
            chronological,
            lambda r: edge_bucket(r.get("_gross_edge"))
        ),
        # Whale-aligned boosts only: separates the bot's whale-trusted
        # bets from its plain-Kelly bets. Both panels drop rows with
        # `_whale_class is None` (i.e. non-whale-boosted trades), so N
        # in these panels = "number of whale-boosted trades".
        # by_whale_class: aggressive / burst / resting / other.
        # by_whale_magnitude: continuous bucket inside the class
        # (e.g. price_7-10c, burst_25-50k) — tells us whether boost
        # SIZE inside a class predicts realized P&L.
        "by_whale_class": _bucket_aggregate(
            chronological, lambda r: r.get("_whale_class")
        ),
        "by_whale_magnitude": _bucket_aggregate(
            chronological, lambda r: r.get("_whale_magnitude")
        ),
        # Side × entry-price bucket — answers "do YES underdogs win more
        # than NO underdogs at the same price?" Drives the eventual
        # asymmetric tuning of min_entry_price_yes / min_entry_price_no.
        # Currently both floors are 0.35 so this panel measures only;
        # doesn't influence policy until we set asymmetric values.
        "by_side_x_entry": _bucket_aggregate(
            chronological,
            lambda r: _side_entry_bucket(r.get("side"), r.get("fill_price"))
        ),
        # Whale-boosted trades only, bucketed by the side WE bet.
        # Tests the hypothesis: are NO-side boosts more predictive than
        # YES-side boosts? In a symmetric two-way market they should be
        # equivalent, but Kalshi retail skews YES-heavy on favorites so
        # NO-side whales may be smarter money fading retail. Empirical
        # answer comes from this panel once we have N>20 boosted trades.
        "by_whale_side": _bucket_aggregate(
            chronological,
            lambda r: (
                f"whale→{r.get('side')}" if r.get("_whale_class") else None
            )
        ),
        # Windowed P&L: realized within each window from chronological
        # (already enriched/cleaned). Unrealized is a separate snapshot
        # because it's window-independent (open positions are open NOW).
        "windowed_pnl": [
            _windowed_pnl(chronological, h) for h in (3, 6, 12, 24)
        ],
        "open_mtm": _open_unrealized_snapshot(_journal.open_positions()),
        # Tennis P&L tracker — the bot's one proven surface, isolated.
        "tennis_summary": _tennis_summary(chronological),
        # ---- CLV summary -----------------------------------------------
        # CLV per trade: clv_price - fill_price (>0 if line moved our way).
        # Aggregated across trades with a non-null clv_price.
        **_clv_summary(chronological),
        # ---- Settlement backtest ---------------------------------------
        # "What if we held to settlement?" — populated by the periodic
        # settlement_backfill task. Tells us whether exits cost us money.
        "settlement": _journal.settlement_summary(),
    }


def _clv_summary(trades: list[dict]) -> dict:
    """Compute CLV stats: avg, win rate, by-sport breakdown."""
    sampled = [
        t for t in trades
        if t.get("clv_price") is not None and t.get("fill_price") is not None
    ]
    if not sampled:
        return {
            "n_clv": 0, "avg_clv_bp": 0.0, "pct_positive_clv": 0.0,
            "by_sport_clv": [],
        }
    deltas = [(t["clv_price"] - t["fill_price"]) for t in sampled]
    n = len(deltas)
    avg = sum(deltas) / n
    positives = sum(1 for d in deltas if d > 0)
    # Aggregate CLV per sport
    by_sport: dict[str, list[float]] = {}
    for t, d in zip(sampled, deltas):
        s = t.get("_sport") or "?"
        by_sport.setdefault(s, []).append(d)
    by_sport_clv = sorted(
        (
            {
                "key": k,
                "n": len(v),
                "avg_clv_bp": round(sum(v) * 100 / len(v), 2),
                "pct_positive": round(100 * sum(1 for d in v if d > 0) / len(v), 1),
            }
            for k, v in by_sport.items()
        ),
        key=lambda r: -r["n"],
    )
    return {
        "n_clv": n,
        "avg_clv_bp": round(avg * 100, 2),
        "pct_positive_clv": round(100 * positives / n, 1),
        "by_sport_clv": by_sport_clv,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    env = env_config()
    cfg = file_config()
    return _render_dashboard(
        mode=env.mode,
        kalshi_env=env.kalshi_env,
        bankroll=cfg.bankroll_usd,
        min_edge_bp=int(cfg.decision.min_edge * 100),
        min_entry_price=cfg.decision.min_entry_price,
    )


# ----- HTML/JS dashboard ----------------------------------------------------

def _render_dashboard(*, mode: str, kalshi_env: str, bankroll: float,
                     min_edge_bp: int, min_entry_price: float) -> str:
    return rf"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Edge Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root {{
  --bg: #0A0F14; --surface: #151A21; --surface-raised: #1E242D;
  --surface-overlay: #252B35; --border: #2A313A;
  --text: #FFFFFF; --text-secondary: #9CA3AF; --text-muted: #5E6470;
  --primary: #3DA5F5; --primary-soft: rgba(61,165,245,0.18);
  --accent: #7C9EFF;
  --success: #3DD68C; --success-soft: rgba(61,214,140,0.15);
  --danger: #FF4D4D; --danger-soft: rgba(255,77,77,0.15);
  --warning: #FBB454; --warning-soft: rgba(251,180,84,0.15);
  --info: #5BC0EB; --info-soft: rgba(91,192,235,0.12);
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, sans-serif; font-size: 15px; line-height: 1.5;
  -webkit-font-smoothing: antialiased; font-variant-numeric: tabular-nums; }}
a {{ color: var(--primary); text-decoration: none; transition: color 0.15s; }}
a:hover {{ color: var(--accent); }}

.shell {{ max-width: 1240px; margin: 0 auto; padding: 32px 24px 80px; }}

.header {{ display: flex; align-items: flex-start; justify-content: space-between;
  flex-wrap: wrap; gap: 16px; margin-bottom: 32px; }}
.brand h1 {{ margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }}
.brand .sub {{ color: var(--text-muted); font-size: 12px; margin-top: 6px;
  display: flex; align-items: center; gap: 8px; }}
.live-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--success);
  box-shadow: 0 0 8px var(--success); animation: pulse 2s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.35}} }}
.badges {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.badge {{ font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  padding: 6px 10px; border-radius: 6px; background: var(--surface); color: var(--text-secondary); }}
.badge.live {{ color: var(--danger); background: var(--danger-soft); }}
.badge.paper {{ color: var(--info); background: var(--info-soft); }}
.badge.prod {{ color: var(--warning); background: var(--warning-soft); }}
.badge.demo {{ color: var(--text-muted); }}

.hero {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 32px; }}
.dial {{ background: var(--surface); border-radius: 16px; padding: 28px 24px;
  display: flex; flex-direction: column; align-items: center; gap: 14px; }}
.dial-svg {{ width: 200px; height: 200px; position: relative; }}
.dial-svg svg {{ transform: rotate(-90deg); }}
.dial-track {{ stroke: var(--surface-raised); fill: none; stroke-width: 10; }}
.dial-arc {{ fill: none; stroke-width: 10; stroke-linecap: round;
  transition: stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.3s; }}
.dial-center {{ position: absolute; inset: 0; display: flex; align-items: center;
  justify-content: center; flex-direction: column; gap: 4px; }}
.dial-value {{ font-size: 42px; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }}
.dial-value.pos {{ color: var(--success); }}
.dial-value.neg {{ color: var(--danger); }}
.dial-unit {{ font-size: 13px; font-weight: 500; color: var(--text-secondary); letter-spacing: 0.04em; }}
.dial-label {{ font-size: 11px; font-weight: 600; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--text-secondary); }}
.dial-sub {{ font-size: 12px; color: var(--text-muted); margin-top: -4px; text-align: center; }}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 24px; }}
.kpi {{ background: var(--surface); border-radius: 16px; padding: 18px 20px; }}
.kpi .label {{ font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-secondary); margin-bottom: 10px; }}
.kpi .value {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }}
.kpi .value.pos {{ color: var(--success); }} .kpi .value.neg {{ color: var(--danger); }}
.kpi .sub {{ font-size: 12px; color: var(--text-muted); margin-top: 8px; }}

.card {{ background: var(--surface); border-radius: 16px; padding: 20px 22px; }}
.card-title {{ display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 16px; gap: 12px; flex-wrap: wrap; }}
.card-title h3 {{ margin: 0; font-size: 11px; font-weight: 600; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--text-secondary); }}
.card-title .meta {{ font-size: 12px; color: var(--text-muted); font-weight: 400;
  letter-spacing: 0; text-transform: none; }}

.section {{ font-size: 11px; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--text-secondary);
  margin: 40px 0 16px; padding-top: 24px; border-top: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px; }}
.section::after {{ content: ''; flex: 1; height: 1px; background: var(--border); margin-left: 8px; }}

.grid {{ display: grid; gap: 16px; }}
.grid.cols-2 {{ grid-template-columns: 1fr 1fr; }}
.grid.cols-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.grid.cols-4 {{ grid-template-columns: repeat(4, 1fr); }}
@media (max-width: 1024px) {{
  .grid.cols-3, .grid.cols-4 {{ grid-template-columns: 1fr 1fr; }}
  .hero {{ grid-template-columns: 1fr; max-width: 480px; margin: 0 auto 32px; }}
}}
@media (max-width: 600px) {{
  .grid.cols-2, .grid.cols-3, .grid.cols-4 {{ grid-template-columns: 1fr; }}
  .shell {{ padding: 20px 14px 60px; }} .dial-value {{ font-size: 36px; }}
}}

/* Phone-preview mode — class-triggered so it works at any viewport.
   Lets you eyeball the iPhone layout from a laptop. CSS media queries
   key off VIEWPORT width, so a class is the only way to force the
   mobile layout in a desktop browser. */
.shell.phone-view {{
  max-width: 414px;
  border: 1px solid var(--border);
  border-radius: 28px;
  padding: 24px 16px 60px;
  margin-top: 16px;
}}
.shell.phone-view .hero,
.shell.phone-view .grid.cols-2,
.shell.phone-view .grid.cols-3,
.shell.phone-view .grid.cols-4 {{ grid-template-columns: 1fr; }}
.shell.phone-view .kpis {{ grid-template-columns: 1fr 1fr; }}
.shell.phone-view .dial-value {{ font-size: 36px; }}
.shell.phone-view .settlement-grid {{ grid-template-columns: 1fr; }}
.shell.phone-view .scroll {{ max-height: 320px; }}

.view-toggle {{
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  padding: 6px 12px; border-radius: 8px; cursor: pointer;
  background: var(--surface-raised); color: var(--text-secondary);
  border: 1px solid var(--border); }}
.view-toggle:hover {{ color: var(--text); background: var(--surface-overlay); }}

.chart-wrap {{ height: 260px; position: relative; }}
.chart-wrap.tall {{ height: 320px; }}

table {{ width: 100%; border-collapse: collapse; }}
th {{ font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--text-muted); padding: 10px 12px; text-align: left;
  border-bottom: 1px solid var(--border); }}
th.num {{ text-align: right; }}
td {{ padding: 12px; font-size: 13px;
  border-bottom: 1px solid var(--surface-raised); white-space: nowrap; }}
td.num {{ text-align: right; font-weight: 500; }}
td.pos {{ color: var(--success); }} td.neg {{ color: var(--danger); }}
td.muted {{ color: var(--text-muted); }}
td.ticker {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); }}
tr:last-child td {{ border-bottom: none; }}
.scroll {{ max-height: 420px; overflow-y: auto;
  scrollbar-width: thin; scrollbar-color: var(--surface-raised) transparent; }}
.scroll::-webkit-scrollbar {{ width: 6px; }}
.scroll::-webkit-scrollbar-track {{ background: transparent; }}
.scroll::-webkit-scrollbar-thumb {{ background: var(--surface-raised); border-radius: 3px; }}

.pill {{ display: inline-flex; align-items: center; gap: 4px;
  font-size: 10px; font-weight: 600; padding: 4px 8px; border-radius: 6px; letter-spacing: 0.04em; }}
.pill.success {{ color: var(--success); background: var(--success-soft); }}
.pill.danger {{ color: var(--danger); background: var(--danger-soft); }}
.pill.warn {{ color: var(--warning); background: var(--warning-soft); }}
.pill.info {{ color: var(--info); background: var(--info-soft); }}

.tag {{ display: inline-block; font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
  padding: 3px 7px; border-radius: 4px; background: var(--surface-raised);
  color: var(--text-secondary); text-transform: uppercase; }}
.tag.tennis {{ color: #B69CFA; background: rgba(157,124,247,0.15); }}
.tag.cricket {{ color: #F5B470; background: rgba(245,158,61,0.15); }}
.tag.mlb {{ color: #6FD0F5; background: rgba(91,192,235,0.15); }}
.tag.hockey {{ color: #9DB5FF; background: rgba(124,158,255,0.15); }}
.tag.nba, .tag.nfl {{ color: #FFA76E; background: rgba(255,138,61,0.15); }}

.empty {{ color: var(--text-muted); text-align: center; padding: 24px 0; font-style: italic; font-size: 13px; }}

.settlement-grid {{ display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 24px; padding: 8px 0 16px; }}
.settlement-grid .item .lbl {{ font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px; }}
.settlement-grid .item .val {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }}
.settlement-grid .item .val.pos {{ color: var(--success); }}
.settlement-grid .item .val.neg {{ color: var(--danger); }}
.settlement-grid .item .sub {{ font-size: 11px; color: var(--text-muted); margin-top: 6px; }}
.stoploss-note {{ font-size: 12px; color: var(--text-secondary);
  padding-top: 12px; border-top: 1px solid var(--surface-raised); }}

.window-pnl-table td:first-child {{ color: var(--text-secondary);
  font-size: 10px; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; }}

.footer {{ margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border);
  font-size: 11px; color: var(--text-muted); text-align: center; }}
.footer a {{ color: var(--text-secondary); margin: 0 8px; font-weight: 500; letter-spacing: 0.04em; }}

.fab {{ position: fixed; bottom: 24px; right: 24px; width: 56px; height: 56px;
  border-radius: 50%; background: var(--primary); color: var(--bg);
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 22px; box-shadow: 0 0 24px rgba(61,165,245,0.4);
  cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; text-decoration: none; }}
.fab:hover {{ transform: translateY(-2px); box-shadow: 0 4px 32px rgba(61,165,245,0.5); color: var(--bg); }}
</style>
</head>
<body>
<div class="shell">

  <header class="header">
    <div class="brand">
      <h1>Edge Monitor</h1>
      <div class="sub"><span class="live-dot"></span><span id="refresh-status">connecting…</span></div>
    </div>
    <div class="badges">
      <span class="badge {('paper' if mode == 'paper' else 'live')}">{mode}</span>
      <span class="badge {('prod' if kalshi_env == 'prod' else 'demo')}">{kalshi_env}</span>
      <span class="badge">bankroll ${bankroll:.0f}</span>
      <span class="badge">min edge {min_edge_bp}bp</span>
      <span class="badge">min entry ${min_entry_price:.2f}</span>
      <button class="view-toggle" id="view-toggle" onclick="toggleView()">💻 Laptop</button>
    </div>
  </header>

  <section class="hero">
    <div class="dial">
      <div class="dial-svg">
        <svg viewBox="0 0 200 200" width="200" height="200">
          <circle class="dial-track" cx="100" cy="100" r="84"/>
          <circle id="arc-pnl" class="dial-arc" cx="100" cy="100" r="84"
            stroke="var(--primary)" stroke-dasharray="528" stroke-dashoffset="528"/>
        </svg>
        <div class="dial-center">
          <div class="dial-value" id="dial-pnl-value">—</div>
          <div class="dial-unit" id="dial-pnl-unit"></div>
        </div>
      </div>
      <div class="dial-label">Total P&amp;L</div>
      <div class="dial-sub" id="dial-pnl-sub">awaiting first trade</div>
    </div>
    <div class="dial">
      <div class="dial-svg">
        <svg viewBox="0 0 200 200" width="200" height="200">
          <circle class="dial-track" cx="100" cy="100" r="84"/>
          <circle id="arc-winrate" class="dial-arc" cx="100" cy="100" r="84"
            stroke="var(--primary)" stroke-dasharray="528" stroke-dashoffset="528"/>
        </svg>
        <div class="dial-center"><div class="dial-value" id="dial-winrate-value">—</div>
          <div class="dial-unit">%</div></div>
      </div>
      <div class="dial-label">Win Rate</div>
      <div class="dial-sub" id="dial-winrate-sub">—</div>
    </div>
    <div class="dial">
      <div class="dial-svg">
        <svg viewBox="0 0 200 200" width="200" height="200">
          <circle class="dial-track" cx="100" cy="100" r="84"/>
          <circle id="arc-positions" class="dial-arc" cx="100" cy="100" r="84"
            stroke="var(--primary)" stroke-dasharray="528" stroke-dashoffset="528"/>
        </svg>
        <div class="dial-center"><div class="dial-value" id="dial-positions-value">—</div>
          <div class="dial-unit">/ 15</div></div>
      </div>
      <div class="dial-label">Open Positions</div>
      <div class="dial-sub" id="dial-positions-sub">capacity 15</div>
    </div>
  </section>

  <section class="kpis">
    <div class="kpi"><div class="label">Best / Worst</div>
      <div class="value" id="kpi-bestworst" style="font-size:18px">—</div>
      <div class="sub">single-trade range</div></div>
    <div class="kpi"><div class="label">Avg Edge</div>
      <div class="value" id="kpi-edge">—</div><div class="sub">predicted bp</div></div>
    <div class="kpi"><div class="label">Avg CLV</div>
      <div class="value" id="kpi-clv">—</div>
      <div class="sub" id="kpi-clv-sub">closing-line value</div></div>
    <div class="kpi"><div class="label">Trades</div>
      <div class="value" id="kpi-trades">—</div>
      <div class="sub" id="kpi-trades-sub"></div></div>
  </section>

  <div class="card" style="margin-bottom: 16px">
    <div class="card-title"><h3>P&amp;L by Window</h3>
      <span class="meta" id="window-pnl-meta">realized + unrealized mtm</span></div>
    <table class="window-pnl-table"><thead><tr>
      <th></th><th class="num">3h</th><th class="num">6h</th><th class="num">12h</th><th class="num">24h</th>
    </tr></thead>
    <tbody id="tbody-window-pnl"><tr><td colspan="5" class="empty">loading…</td></tr></tbody></table>
  </div>

  <div class="card" style="margin-bottom: 16px">
    <div class="card-title"><h3>Tennis P&amp;L</h3>
      <span class="meta">the bot's proven surface — ATP + WTA isolated</span></div>
    <table class="window-pnl-table"><thead><tr>
      <th></th><th class="num">Tennis</th><th class="num">WTA</th><th class="num">ATP</th>
      <th class="num">24h</th><th class="num">72h</th>
    </tr></thead>
    <tbody id="tbody-tennis"><tr><td colspan="6" class="empty">loading…</td></tr></tbody></table>
  </div>

  <div class="card" style="margin-bottom: 16px">
    <div class="card-title"><h3>Cumulative P&amp;L</h3><span class="meta" id="curve-meta"></span></div>
    <div class="chart-wrap tall"><canvas id="chart-curve"></canvas></div>
  </div>

  <div class="grid cols-2">
    <div class="card">
      <div class="card-title"><h3>Edge Calibration</h3>
        <span class="meta">predicted vs realized $ per trade</span></div>
      <div class="chart-wrap"><canvas id="chart-calibration"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title"><h3>Open Positions</h3><span class="meta" id="open-count"></span></div>
      <div class="scroll"><table><thead><tr>
        <th>Bet</th><th>Sport</th><th>Side</th><th class="num">Size</th>
        <th class="num">Fill</th><th class="num">Now</th><th class="num">P&amp;L</th>
        <th class="num">Edge</th><th>Tip</th><th>Provider</th><th class="num">Held</th>
      </tr></thead>
      <tbody id="tbody-open"><tr><td colspan="11" class="empty">no positions</td></tr></tbody></table></div>
    </div>
  </div>

  <h2 class="section">What works, what doesn't</h2>

  <div class="grid cols-3">
    <div class="card"><div class="card-title"><h3>By Sport</h3></div>
      <table><thead><tr><th>Sport</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-sport"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>Favorite vs Underdog</h3><span class="meta">entry ≥ 50¢</span></div>
      <table><thead><tr><th>Side</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-favdog"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>Home vs Away</h3></div>
      <table><thead><tr><th>Side</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-side"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
  </div>

  <div class="grid cols-3" style="margin-top: 16px">
    <div class="card"><div class="card-title"><h3>By Provider</h3><span class="meta">multi-book or single?</span></div>
      <table><thead><tr><th>Provider</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-provider"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>By Entry Price</h3></div>
      <table><thead><tr><th>Price</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-entry"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>By Time-to-Tip</h3></div>
      <table><thead><tr><th>Window</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-tip"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
  </div>

  <div class="grid cols-2" style="margin-top: 16px">
    <div class="card"><div class="card-title"><h3>By Exit Reason</h3></div>
      <table><thead><tr><th>Reason</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-exit"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>By Series</h3></div>
      <div class="scroll"><table><thead><tr><th>Series</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-series"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div></div>
  </div>

  <div class="card" style="margin-top: 16px">
    <div class="card-title"><h3>Settlement Backtest</h3>
      <span class="meta">would we have made more holding to game end?</span></div>
    <div class="settlement-grid">
      <div class="item"><div class="lbl">Actual P&amp;L</div>
        <div class="val" id="bt-actual">—</div><div class="sub">with exits</div></div>
      <div class="item"><div class="lbl">If Held to Settlement</div>
        <div class="val" id="bt-settle">—</div><div class="sub">counterfactual</div></div>
      <div class="item"><div class="lbl">Delta</div>
        <div class="val" id="bt-delta">—</div><div class="sub" id="bt-delta-sub"></div></div>
    </div>
    <div class="stoploss-note" id="bt-stoploss"></div>
  </div>

  <div class="grid cols-2" style="margin-top: 16px">
    <div class="card"><div class="card-title"><h3>By Hold Duration</h3>
      <span class="meta">exiting too early on stop_loss noise?</span></div>
      <table><thead><tr><th>Hold</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-hold"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>By Model Confidence</h3>
      <span class="meta">calibration check</span></div>
      <table><thead><tr><th>p_yes</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-confidence"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
  </div>

  <div class="grid cols-2" style="margin-top: 16px">
    <div class="card"><div class="card-title"><h3>A/B Exit Cohorts</h3>
      <span class="meta">what if every trade used a different exit?</span></div>
      <table><thead><tr><th>Policy</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-exit-policy"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>By Edge Bucket</h3>
      <span class="meta">bigger edge → bigger P&amp;L?</span></div>
      <table><thead><tr><th>Edge</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-edge-bucket"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table></div>
  </div>

  <div class="grid cols-2" style="margin-top: 16px">
    <div class="card"><div class="card-title"><h3>By Whale Class</h3>
      <span class="meta">aggressive vs burst vs resting</span></div>
      <table><thead><tr><th>Class</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-whale-class"><tr><td colspan="4" class="empty">no whale-boosted trades</td></tr></tbody></table></div>
    <div class="card"><div class="card-title"><h3>By Whale Magnitude</h3>
      <span class="meta">bigger signal → bigger P&amp;L?</span></div>
      <table><thead><tr><th>Bucket</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-whale-magnitude"><tr><td colspan="4" class="empty">no whale-boosted trades</td></tr></tbody></table></div>
  </div>

  <div class="card" style="margin-top: 16px">
    <div class="card-title"><h3>By Whale Side</h3>
      <span class="meta">are NO-side whales smarter money?</span></div>
    <table><thead><tr><th>Side we bet</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
    <tbody id="tbody-whale-side"><tr><td colspan="4" class="empty">no whale-boosted trades</td></tr></tbody></table>
  </div>

  <div class="card" style="margin-top: 16px">
    <div class="card-title"><h3>By Side × Entry Price</h3>
      <span class="meta">YES underdogs vs NO underdogs at same price?</span></div>
    <table><thead><tr><th>Side × Price</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
    <tbody id="tbody-side-x-entry"><tr><td colspan="4" class="empty">no data</td></tr></tbody></table>
  </div>

  <div class="card" style="margin-top: 16px">
    <div class="card-title"><h3>Kalshi vs Polymarket Spreads</h3>
      <span class="meta" id="cx-meta">awaiting first snapshot</span></div>
    <div class="scroll"><table><thead><tr>
      <th>Kalshi market</th><th>Polymarket question</th>
      <th class="num">Kalshi YES</th><th class="num">Poly YES</th>
      <th class="num">Spread</th><th class="num">Match</th><th>Direction</th>
    </tr></thead>
    <tbody id="tbody-cross-exchange"><tr><td colspan="7" class="empty">awaiting first snapshot</td></tr></tbody></table></div>
  </div>

  <div class="card" style="margin-top: 16px">
    <div class="card-title"><h3>CLV by Sport</h3>
      <span class="meta">did the line move our way after we entered?</span></div>
    <table><thead><tr><th>Sport</th><th class="num">N</th><th class="num">Avg CLV</th><th class="num">% Positive</th></tr></thead>
    <tbody id="tbody-clv-sport"><tr><td colspan="4" class="empty">awaiting tipoff samples</td></tr></tbody></table>
  </div>

  <h2 class="section">Golf Advisor · beta</h2>

  <div class="grid cols-2">
    <div class="card">
      <div class="card-title"><h3>Round-Leader Alerts</h3>
        <span class="meta" id="golf-leader-meta">live in-play · DataGolf vs DraftKings</span></div>
      <div class="scroll">
        <table><thead><tr>
          <th>Golfer</th><th>Pos</th><th class="num">Thru</th>
          <th>Market</th><th class="num">DK</th><th class="num">DataGolf</th><th class="num">Edge</th>
        </tr></thead>
        <tbody id="tbody-golf-leader"><tr><td colspan="7" class="empty">no qualifying golfers</td></tr></tbody></table>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><h3>3-Ball / Matchup Edges</h3>
        <span class="meta" id="golf-3ball-meta">DataGolf vs DraftKings matchup lines</span></div>
      <div class="scroll">
        <table><thead><tr>
          <th>Pick</th><th>Event</th><th class="num">DK</th>
          <th class="num">DataGolf</th><th class="num">Edge</th>
        </tr></thead>
        <tbody id="tbody-golf-3ball"><tr><td colspan="5" class="empty">no +EV legs</td></tr></tbody></table>
      </div>
    </div>
  </div>

  <h2 class="section">Recent activity</h2>

  <div class="card">
    <div class="card-title"><h3>Recent Trades</h3><span class="meta" id="trades-count"></span></div>
    <div class="scroll"><table><thead><tr>
      <th>Bet</th><th>Sport</th><th>Side</th><th class="num">Size</th>
      <th class="num">Edge</th><th class="num">Fill</th><th class="num">Exit</th>
      <th class="num">CLV</th><th class="num">P&amp;L</th><th class="num">Fees</th>
      <th>Opened</th><th class="num">Held</th><th>Tip</th><th>Provider</th><th>Why</th>
    </tr></thead>
    <tbody id="tbody-trades"><tr><td colspan="15" class="empty">loading…</td></tr></tbody></table></div>
  </div>

  <div class="card" style="margin-top: 16px">
    <div class="card-title"><h3>Daily P&amp;L</h3></div>
    <table><thead><tr><th>Date</th><th class="num">Trades</th><th class="num">P&amp;L</th></tr></thead>
    <tbody id="tbody-daily"><tr><td colspan="3" class="empty">loading…</td></tr></tbody></table>
  </div>

  <footer class="footer">
    Auto-refreshes every 15s ·
    <a href="/stats">/stats</a> · <a href="/trades">/trades</a> ·
    <a href="/positions">/positions</a> · <a href="/pnl">/pnl</a> ·
    <a href="/pnl_digest">/pnl_digest</a> · <a href="/cross_exchange">/cross_exchange</a>
  </footer>
</div>

<a href="/pnl_digest?send=true" target="_blank" class="fab" title="Send P&L digest to Slack">⚡</a>

<script>
const fmt$ = n => (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
const fmtTime = iso => {{ if (!iso) return ''; try {{ return new Date(iso).toLocaleString(undefined, {{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}}); }} catch {{ return iso; }} }};
const fmtClock = iso => {{ if (!iso) return ''; try {{ return new Date(iso).toLocaleTimeString(undefined, {{hour:'2-digit',minute:'2-digit'}}); }} catch {{ return ''; }} }};
const cls = n => n > 0 ? 'pos' : (n < 0 ? 'neg' : 'muted');
const sportFromTicker = t => {{
  const s = (t || '').split('-')[0].toUpperCase();
  return ({{KXNBAGAME:'NBA',KXNFLGAME:'NFL',KXMLBGAME:'MLB',KXNHLGAME:'NHL',KXATPMATCH:'ATP',KXWTAMATCH:'WTA',KXIPLGAME:'IPL',KXCRICKETTESTMATCH:'TEST',KXWNBAGAME:'WNBA',KXPGATOUR:'PGA'}})[s] || s;
}};
const sportTagClass = code => {{
  const c = (code || '').toLowerCase();
  if (c === 'atp' || c === 'wta') return 'tag tennis';
  if (c === 'ipl' || c === 'test') return 'tag cricket';
  if (c === 'mlb') return 'tag mlb';
  if (c === 'nhl') return 'tag hockey';
  if (c === 'nba' || c === 'wnba' || c === 'nfl') return 'tag nba';
  return 'tag';
}};
const fmtHeld = (openIso, closeIso) => {{
  if (!openIso || !closeIso) return '';
  const ms = new Date(closeIso) - new Date(openIso);
  if (!isFinite(ms) || ms < 0) return '';
  const mins = Math.round(ms / 60000);
  if (mins < 60) return mins + 'm';
  const h = Math.floor(mins / 60), m = mins % 60;
  return m === 0 ? `${{h}}h` : `${{h}}h ${{m}}m`;
}};
const parseReason = reason => {{
  if (!reason) return {{}};
  const out = {{}}; let m;
  if ((m = reason.match(/book\[([^\]]+)\]/))) {{
    const provider = m[1];
    if (provider.startsWith('pinnacle')) out.provider = provider.includes(':') ? provider : 'pinnacle';
    else if (provider.startsWith('odds_api')) {{ out.provider = 'odds_api';
      const bm = provider.match(/(\d+)\s+books/); if (bm) out.books = bm[1]; }}
    else if (provider.startsWith('espn')) out.provider = 'espn';
    else if (provider.startsWith('datagolf')) out.provider = 'datagolf';
    else out.provider = provider;
  }}
  if ((m = reason.match(/tip in (-?\d+)min/)) || (m = reason.match(/start in (-?\d+)min/))) out.minsToTip = parseInt(m[1]);
  if ((m = reason.match(/pregame\s+(.+?)\s+\|/))) out.matchup = m[1].trim();
  if ((m = reason.match(/our=([^()|]+?)(?:\(|\s+p_yes|\s*\|)/))) out.ourSide = m[1].trim();
  if ((m = reason.match(/our=[^()|]+\((home|away)\)/))) out.ourLoc = m[1];
  if ((m = reason.match(/p_yes=([\d.]+)/))) out.pYes = parseFloat(m[1]);
  return out;
}};
function feePerContract(price) {{ const p = Math.max(0.01, Math.min(0.99, price));
  return Math.ceil(0.07 * p * (1 - p) * 100) / 100; }}

let curveChart, calibChart;
function buildCurveChart(curve) {{
  const ctx = document.getElementById('chart-curve');
  const labels = curve.map(p => fmtTime(p.ts));
  const values = curve.map(p => p.cum);
  if (curveChart) {{ curveChart.data.labels = labels; curveChart.data.datasets[0].data = values; curveChart.update('none'); return; }}
  curveChart = new Chart(ctx, {{ type: 'line',
    data: {{ labels, datasets: [{{ label: 'Cumulative P&L', data: values,
      borderColor: '#3DA5F5', backgroundColor: 'rgba(61,165,245,0.08)',
      borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.18, fill: true }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ backgroundColor: '#1E242D', borderColor: '#2A313A', borderWidth: 1,
          titleColor: '#FFF', bodyColor: '#9CA3AF', padding: 10, cornerRadius: 8, displayColors: false,
          callbacks: {{ title: i => i[0].label, label: i => '$' + i.parsed.y.toFixed(2) }} }} }},
      scales: {{ x: {{ ticks: {{ color: '#5E6470', font: {{ size: 10 }} }}, grid: {{ color: 'transparent' }} }},
        y: {{ ticks: {{ color: '#5E6470', font: {{ size: 10 }}, callback: v => '$' + v.toFixed(0) }},
          grid: {{ color: 'rgba(42,49,58,0.5)', drawBorder: false }} }} }} }} }});
}}
function buildCalibChart(rows) {{
  const ctx = document.getElementById('chart-calibration');
  const labels = rows.map(r => r.bucket_pp + 'bp');
  const predicted = rows.map(r => r.predicted);
  const realized = rows.map(r => r.realized);
  if (calibChart) {{ calibChart.data.labels = labels; calibChart.data.datasets[0].data = predicted; calibChart.data.datasets[1].data = realized; calibChart.update('none'); return; }}
  calibChart = new Chart(ctx, {{ type: 'bar',
    data: {{ labels, datasets: [
      {{ label: 'Predicted', data: predicted, backgroundColor: 'rgba(61,165,245,0.6)', borderRadius: 4 }},
      {{ label: 'Realized', data: realized, backgroundColor: 'rgba(61,214,140,0.6)', borderRadius: 4 }}, ] }},
    options: {{ responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ color: '#9CA3AF', font: {{ size: 11 }} }} }},
        tooltip: {{ backgroundColor: '#1E242D', borderColor: '#2A313A', borderWidth: 1,
          titleColor: '#FFF', bodyColor: '#9CA3AF', padding: 10, cornerRadius: 8 }} }},
      scales: {{ x: {{ ticks: {{ color: '#5E6470', font: {{ size: 10 }} }}, grid: {{ color: 'transparent' }} }},
        y: {{ ticks: {{ color: '#5E6470', font: {{ size: 10 }}, callback: v => '$' + v.toFixed(0) }},
          grid: {{ color: 'rgba(42,49,58,0.5)', drawBorder: false }} }} }} }} }});
}}

function setDial(arcId, valueId, unitId, subId, value, opts) {{
  const DASH = 528;
  const arc = document.getElementById(arcId);
  const val = document.getElementById(valueId);
  const unit = unitId ? document.getElementById(unitId) : null;
  const sub  = subId  ? document.getElementById(subId)  : null;
  if (opts.format === 'money') {{
    val.textContent = (value >= 0 ? '+$' : '-$') + Math.abs(value).toFixed(2);
    val.className = 'dial-value ' + (value > 0 ? 'pos' : value < 0 ? 'neg' : '');
    arc.setAttribute('stroke', value > 0 ? 'var(--success)' : value < 0 ? 'var(--danger)' : 'var(--primary)');
  }} else if (opts.format === 'percent') {{
    val.textContent = (value * 100).toFixed(0); val.className = 'dial-value';
    arc.setAttribute('stroke', value >= 0.55 ? 'var(--success)' : value >= 0.45 ? 'var(--primary)' : 'var(--danger)');
  }} else {{
    val.textContent = value; val.className = 'dial-value';
    arc.setAttribute('stroke', 'var(--primary)');
  }}
  if (unit && opts.unit !== undefined) unit.textContent = opts.unit;
  if (sub && opts.sub !== undefined) sub.textContent = opts.sub;
  const ratio = Math.max(0, Math.min(1, opts.fillRatio || 0));
  arc.setAttribute('stroke-dashoffset', DASH * (1 - ratio));
}}

function updateDials(stats) {{
  const n = stats.n_trades || 0;
  const pnl = stats.total_pnl || 0;
  const wr = stats.win_rate || 0;
  const open = stats.open_positions || 0;
  setDial('arc-pnl', 'dial-pnl-value', 'dial-pnl-unit', 'dial-pnl-sub', pnl, {{
    format: 'money', unit: '', fillRatio: Math.min(Math.abs(pnl) / 200, 1),
    sub: n > 0 ? `${{n}} trades · avg ${{fmt$(stats.avg_pnl||0)}}/trade` : 'awaiting first trade' }});
  setDial('arc-winrate', 'dial-winrate-value', null, 'dial-winrate-sub', wr, {{
    format: 'percent', fillRatio: wr,
    sub: n > 0 ? `${{stats.n_wins||0}}W · ${{stats.n_losses||0}}L` : '—' }});
  setDial('arc-positions', 'dial-positions-value', null, 'dial-positions-sub', open, {{
    format: 'count', fillRatio: Math.min(open / 15, 1), sub: 'capacity 15' }});
}}

function renderKpis(stats) {{
  document.getElementById('kpi-bestworst').innerHTML =
    `<span class="pos">${{fmt$(stats.best_trade||0)}}</span> / <span class="neg">${{fmt$(stats.worst_trade||0)}}</span>`;
  document.getElementById('kpi-edge').textContent = (stats.avg_edge_bp||0).toFixed(1) + 'bp';
  const clv = stats.avg_clv_bp || 0;
  const clvEl = document.getElementById('kpi-clv');
  if (stats.n_clv > 0) {{
    clvEl.className = 'value ' + cls(clv);
    clvEl.textContent = (clv >= 0 ? '+' : '') + clv.toFixed(1) + 'bp';
    document.getElementById('kpi-clv-sub').textContent = `${{stats.n_clv}} samples · ${{(stats.pct_positive_clv||0).toFixed(0)}}% positive`;
  }} else {{
    clvEl.textContent = '—';
    document.getElementById('kpi-clv-sub').textContent = 'awaiting tipoff samples';
  }}
  document.getElementById('kpi-trades').textContent = stats.n_trades || 0;
  document.getElementById('kpi-trades-sub').textContent =
    `${{stats.n_wins||0}}W · ${{stats.n_losses||0}}L · ${{((stats.win_rate||0)*100).toFixed(0)}}% wr`;
}}

function renderBucketTable(tbodyId, rows) {{
  const tb = document.getElementById(tbodyId);
  if (!rows || !rows.length) {{ tb.innerHTML = '<tr><td colspan="4" class="empty">no data yet</td></tr>'; return; }}
  tb.innerHTML = rows.map(r => `<tr>
    <td>${{r.key}}</td><td class="num">${{r.n}}</td>
    <td class="num">${{(r.win_rate*100).toFixed(0)}}%</td>
    <td class="num ${{cls(r.pnl)}}">${{fmt$(r.pnl)}}</td></tr>`).join('');
}}
function renderClvTable(tbodyId, rows) {{
  const tb = document.getElementById(tbodyId);
  if (!rows || !rows.length) {{ tb.innerHTML = '<tr><td colspan="4" class="empty">awaiting tipoff samples</td></tr>'; return; }}
  tb.innerHTML = rows.map(r => `<tr>
    <td>${{r.key}}</td><td class="num">${{r.n}}</td>
    <td class="num ${{cls(r.avg_clv_bp)}}">${{(r.avg_clv_bp>=0?'+':'')}}${{r.avg_clv_bp.toFixed(1)}}bp</td>
    <td class="num">${{r.pct_positive.toFixed(0)}}%</td></tr>`).join('');
}}

function renderTennis(ts) {{
  const tb = document.getElementById('tbody-tennis');
  if (!ts || !ts.lifetime) {{
    tb.innerHTML = '<tr><td colspan="6" class="empty">no tennis trades yet</td></tr>'; return;
  }}
  const cols = [ts.lifetime, ts.wta, ts.atp, ts.last_24h, ts.last_72h];
  const cell = (c, fn) => fn(c);
  const rows = [
    {{ label: 'Trades', fn: c => `<span>${{c.n}}</span>` }},
    {{ label: 'W / L', fn: c => `<span class="muted">${{c.wins}}/${{c.losses}}</span>` }},
    {{ label: 'Win%', fn: c => `<span>${{(c.win_rate*100).toFixed(0)}}%</span>` }},
    {{ label: 'P&L', fn: c => `<span class="${{cls(c.pnl)}}">${{fmt$(c.pnl)}}</span>` }},
  ];
  tb.innerHTML = rows.map(r => `<tr><td>${{r.label}}</td>` +
    cols.map(c => `<td class="num">${{r.fn(c)}}</td>`).join('') + `</tr>`).join('');
}}

function renderWindowedPnl(windows, openMtm) {{
  const tb = document.getElementById('tbody-window-pnl');
  const meta = document.getElementById('window-pnl-meta');
  if (!windows || !windows.length) {{
    tb.innerHTML = '<tr><td colspan="5" class="empty">no closed trades yet</td></tr>';
    meta.textContent = openMtm && openMtm.n_open_total > 0
      ? `${{openMtm.n_open_marked}}/${{openMtm.n_open_total}} marked · ${{fmt$(openMtm.unrealized_usd||0)}} unrealized`
      : 'awaiting trades'; return;
  }}
  const unr = openMtm ? openMtm.unrealized_usd : 0;
  const nMarked = openMtm ? openMtm.n_open_marked : 0;
  const nTotal = openMtm ? openMtm.n_open_total : 0;
  meta.textContent = nTotal ? `${{nMarked}}/${{nTotal}} open positions marked-to-market` : 'no open positions';
  const unrealCell = `<span class="${{cls(unr)}}">${{fmt$(unr)}}</span>`;
  const rows = [
    {{ label: 'Closed N', fn: w => `<span>${{w.n}}</span>` }},
    {{ label: 'W / L', fn: w => `<span class="muted">${{w.wins}}/${{w.losses}}</span>` }},
    {{ label: 'Realized', fn: w => `<span class="${{cls(w.realized_usd)}}">${{fmt$(w.realized_usd)}}</span>` }},
    {{ label: 'Unrealized', fn: _w => unrealCell }},
    {{ label: 'Net', fn: w => {{ const net = (w.realized_usd || 0) + unr;
      return `<span class="${{cls(net)}}">${{fmt$(net)}}</span>`; }} }},
  ];
  tb.innerHTML = rows.map(r => `<tr><td>${{r.label}}</td>
    <td class="num">${{r.fn(windows[0])}}</td><td class="num">${{r.fn(windows[1])}}</td>
    <td class="num">${{r.fn(windows[2])}}</td><td class="num">${{r.fn(windows[3])}}</td></tr>`).join('');
}}

function renderCrossExchange(snap) {{
  const tb = document.getElementById('tbody-cross-exchange');
  const meta = document.getElementById('cx-meta');
  if (!snap || !snap.spreads || !snap.spreads.length) {{
    tb.innerHTML = '<tr><td colspan="7" class="empty">no spreads found</td></tr>';
    meta.textContent = snap && snap.ts
      ? `last snapshot ${{fmtTime(snap.ts)}} — Kalshi ${{snap.n_kalshi}} / Polymarket ${{snap.n_polymarket}}`
      : 'awaiting first snapshot'; return;
  }}
  meta.textContent = `last snapshot ${{fmtTime(snap.ts)}} — Kalshi ${{snap.n_kalshi}} / Polymarket ${{snap.n_polymarket}} — ${{snap.spreads.length}} spreads`;
  tb.innerHTML = snap.spreads.map(s => {{
    const sp = s.spread_pp; const pp = (sp >= 0 ? '+' : '') + (sp * 100).toFixed(1) + 'pp';
    return `<tr>
      <td class="ticker" title="${{s.kalshi_ticker}}">${{s.kalshi_title || s.kalshi_ticker}}</td>
      <td title="${{s.polymarket_slug}}">${{s.polymarket_question}}</td>
      <td class="num">${{(s.kalshi_yes_price * 100).toFixed(0)}}¢</td>
      <td class="num">${{(s.polymarket_yes_price * 100).toFixed(0)}}¢</td>
      <td class="num ${{cls(sp)}}">${{pp}}</td>
      <td class="num">${{(s.match_score * 100).toFixed(0)}}%</td>
      <td class="muted">${{(s.arb_direction || '').replace(/_/g,' ').replace('buy ','')}}</td>
    </tr>`;
  }}).join('');
}}

function renderOpen(positions) {{
  document.getElementById('open-count').textContent = positions.length + ' open';
  const tb = document.getElementById('tbody-open');
  if (!positions.length) {{ tb.innerHTML = '<tr><td colspan="11" class="empty">no positions</td></tr>'; return; }}
  tb.innerHTML = positions.map(p => {{
    const meta = parseReason(p.reason);
    const sport = sportFromTicker(p.ticker);
    const sportTag = `<span class="${{sportTagClass(sport)}}">${{sport}}</span>`;
    const sideTag = p.side === 'yes' ? `<span class="pill info">YES</span>` : `<span class="pill warn">NO</span>`;
    const fill = (p.fill_price || 0);
    const mid = p.current_mid;
    const nowCell = (mid != null) ? `<span class="${{cls(mid - fill)}}">${{(mid * 100).toFixed(0)}}¢</span>` : '<span class="muted">—</span>';
    let pnlNow = '<span class="muted">—</span>';
    if (mid != null) {{
      const contracts = (p.size_usd || 0) / Math.max(fill, 0.01);
      const grossPnl = (mid - fill) * contracts;
      const fee = feePerContract(mid) * contracts;
      const net = grossPnl - fee;
      pnlNow = `<span class="${{cls(net)}}">${{fmt$(net)}}</span>`;
    }}
    const tip = (meta.minsToTip != null)
      ? (meta.minsToTip < 60 ? meta.minsToTip + 'm' : (meta.minsToTip/60).toFixed(1)+'h')
      : ((p.reason || '').includes('late-game') ? '<span class="muted">live</span>' : '<span class="muted">—</span>');
    const provider = meta.provider || '—';
    const held = fmtHeld(p.opened_ts, new Date().toISOString());
    let betLabel = meta.ourSide || p.ticker.split('-').pop() || '?';
    if (meta.ourLoc) betLabel += ` <span class="muted">(${{meta.ourLoc}})</span>`;
    if (meta.matchup) betLabel += `<br><span class="muted" style="font-size:11px">${{meta.matchup}}</span>`;
    return `<tr><td>${{betLabel}}</td><td>${{sportTag}}</td><td>${{sideTag}}</td>
      <td class="num">$${{(p.size_usd||0).toFixed(0)}}</td>
      <td class="num">${{(fill*100).toFixed(0)}}¢</td>
      <td class="num">${{nowCell}}</td><td class="num">${{pnlNow}}</td>
      <td class="num muted">${{((p.edge||0)*100).toFixed(1)}}bp</td>
      <td>${{tip}}</td><td class="muted">${{provider}}</td>
      <td class="num muted">${{held}}</td></tr>`;
  }}).join('');
}}

function renderTrades(rows) {{
  const tb = document.getElementById('tbody-trades');
  const closed = rows.filter(r => r.closed_ts && r.pnl_usd != null);
  document.getElementById('trades-count').textContent = `${{closed.length}} closed`;
  if (!closed.length) {{ tb.innerHTML = '<tr><td colspan="15" class="empty">no closed trades</td></tr>'; return; }}
  tb.innerHTML = closed.map(r => {{
    const meta = parseReason(r.reason);
    const sport = sportFromTicker(r.ticker);
    const sportTag = `<span class="${{sportTagClass(sport)}}">${{sport}}</span>`;
    const sideTag = r.side === 'yes' ? `<span class="pill info">YES</span>` : `<span class="pill warn">NO</span>`;
    const clvDelta = (r.clv_price != null && r.fill_price != null) ? (r.clv_price - r.fill_price) : null;
    const clvCell = clvDelta != null
      ? `<span class="${{cls(clvDelta)}}">${{(clvDelta*100>=0?'+':'')}}${{(clvDelta*100).toFixed(1)}}bp</span>`
      : '<span class="muted">—</span>';
    const tipCell = (meta.minsToTip != null)
      ? (meta.minsToTip < 60 ? meta.minsToTip + 'm' : (meta.minsToTip/60).toFixed(1)+'h')
      : ((r.reason || '').includes('late-game') ? '<span class="muted">live</span>' : '<span class="muted">—</span>');
    let betLabel = meta.ourSide || r.ticker.split('-').pop() || '?';
    if (meta.ourLoc) betLabel += ` <span class="muted">(${{meta.ourLoc}})</span>`;
    if (meta.matchup) betLabel += `<br><span class="muted" style="font-size:11px">${{meta.matchup}}</span>`;
    return `<tr><td>${{betLabel}}</td><td>${{sportTag}}</td><td>${{sideTag}}</td>
      <td class="num">$${{(r.size_usd||0).toFixed(0)}}</td>
      <td class="num muted">${{((r.edge||0)*100).toFixed(1)}}bp</td>
      <td class="num">${{((r.fill_price||0)*100).toFixed(0)}}¢</td>
      <td class="num">${{((r.exit_price||0)*100).toFixed(0)}}¢</td>
      <td class="num">${{clvCell}}</td>
      <td class="num ${{cls(r.pnl_usd)}}">${{fmt$(r.pnl_usd)}}</td>
      <td class="num muted">$${{(r.fees_usd||0).toFixed(2)}}</td>
      <td class="muted">${{fmtClock(r.opened_ts)}}</td>
      <td class="num muted">${{fmtHeld(r.opened_ts, r.closed_ts)}}</td>
      <td>${{tipCell}}</td><td class="muted">${{meta.provider || '—'}}</td>
      <td class="muted">${{r.exit_reason || ''}}</td></tr>`;
  }}).join('');
}}

function renderDaily(daily) {{
  const tb = document.getElementById('tbody-daily');
  const dates = Object.keys(daily).sort().reverse();
  if (!dates.length) {{ tb.innerHTML = '<tr><td colspan="3" class="empty">no daily data</td></tr>'; return; }}
  tb.innerHTML = dates.map(d => {{ const row = daily[d];
    return `<tr><td>${{d}}</td><td class="num">${{row.n}}</td><td class="num ${{cls(row.pnl)}}">${{fmt$(row.pnl)}}</td></tr>`;
  }}).join('');
}}

function renderBacktest(bt) {{
  if (!bt || bt.n_resolved === 0) {{
    document.getElementById('bt-actual').textContent = '—';
    document.getElementById('bt-settle').textContent = '—';
    document.getElementById('bt-delta').textContent = '—';
    document.getElementById('bt-delta-sub').textContent = 'awaiting Kalshi resolutions';
    document.getElementById('bt-stoploss').textContent = ''; return;
  }}
  const a = document.getElementById('bt-actual');
  a.textContent = fmt$(bt.actual_total); a.className = 'val ' + cls(bt.actual_total);
  const s = document.getElementById('bt-settle');
  s.textContent = fmt$(bt.settlement_total); s.className = 'val ' + cls(bt.settlement_total);
  const d = document.getElementById('bt-delta');
  d.textContent = fmt$(bt.delta); d.className = 'val ' + cls(bt.delta);
  document.getElementById('bt-delta-sub').textContent =
    `${{bt.n_resolved}} resolved · holding beat exit on ${{bt.better_held_pct}}%`;
  if (bt.stop_loss_n > 0) {{
    const slDelta = bt.stop_loss_settlement - bt.stop_loss_actual;
    const stopVerdict = -slDelta;
    const verdictWord = stopVerdict >= 0 ? 'saved' : 'cost';
    document.getElementById('bt-stoploss').innerHTML =
      `Of ${{bt.stop_loss_n}} stop_loss exits: actual ${{fmt$(bt.stop_loss_actual)}}, ` +
      `if held to settlement ${{fmt$(bt.stop_loss_settlement)}} — ` +
      `<span class="${{cls(stopVerdict)}}">stops ${{verdictWord}} ${{fmt$(Math.abs(stopVerdict))}}</span>`;
  }} else {{ document.getElementById('bt-stoploss').textContent = ''; }}
}}

// Golf round-leader alerts
function renderGolfLeader(data) {{
  const tb = document.getElementById('tbody-golf-leader');
  const meta = document.getElementById('golf-leader-meta');
  const alerts = (data && data.alerts) || [];
  if (!alerts.length) {{
    tb.innerHTML = '<tr><td colspan="7" class="empty">no qualifying golfers right now</td></tr>';
    meta.textContent = 'live in-play · no near-leader with soft DK odds';
    return;
  }}
  meta.textContent = `${{alerts.length}} golfer(s) · DataGolf vs DraftKings`;
  const rows = [];
  alerts.forEach(a => {{
    const pos = a.strokes_back === 0 ? 'LEADING' : a.strokes_back + ' back';
    const me = a.market_edges || {{}};
    const markets = Object.keys(me);
    if (!markets.length) {{
      rows.push(`<tr><td>${{a.player}}</td><td>${{a.pos}}</td>
        <td class="num">${{a.thru}}</td><td colspan="4" class="muted">—</td></tr>`);
      return;
    }}
    markets.forEach((mkt, i) => {{
      const [model, dk, edge] = me[mkt];
      const first = i === 0;
      rows.push(`<tr>
        <td>${{first ? '<b>'+a.player+'</b>' : ''}}</td>
        <td class="muted">${{first ? pos+' ('+a.pos+')' : ''}}</td>
        <td class="num">${{first ? 'R'+a.round+' · '+a.thru : ''}}</td>
        <td>${{mkt}}</td>
        <td class="num">${{(dk*100).toFixed(1)}}%</td>
        <td class="num">${{(model*100).toFixed(1)}}%</td>
        <td class="num ${{cls(edge)}}">${{(edge*100>=0?'+':'')}}${{(edge*100).toFixed(1)}}pp</td>
      </tr>`);
    }});
  }});
  tb.innerHTML = rows.join('');
}}

// Golf 3-ball / matchup edges
function renderGolf3ball(data) {{
  const tb = document.getElementById('tbody-golf-3ball');
  const meta = document.getElementById('golf-3ball-meta');
  const edges = (data && data.edges) || [];
  if (!edges.length) {{
    tb.innerHTML = '<tr><td colspan="5" class="empty">no +EV legs right now</td></tr>';
    meta.textContent = 'DataGolf vs DraftKings · no matchup edges';
    return;
  }}
  meta.textContent = `${{edges.length}} +EV leg(s) flagged`;
  tb.innerHTML = edges.map(e => {{
    const dog = e.is_underdog ? ' 🐕' : '';
    return `<tr>
      <td><b>${{e.pick}}</b>${{dog}}</td>
      <td class="muted">${{e.event_name}} R${{e.round_num}}</td>
      <td class="num">${{(e.dk_prob*100).toFixed(0)}}%</td>
      <td class="num">${{(e.datagolf_prob*100).toFixed(0)}}%</td>
      <td class="num pos">+${{(e.edge_pp*100).toFixed(1)}}pp</td>
    </tr>`;
  }}).join('');
}}

async function refresh() {{
  try {{
    const [stats, openPos, tradesRows, daily, crossEx, golfLeader, golf3ball] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/positions').then(r => r.json()),
      fetch('/trades?limit=200').then(r => r.json()),
      fetch('/pnl').then(r => r.json()),
      fetch('/cross_exchange').then(r => r.json()),
      fetch('/golf_leader').then(r => r.json()).catch(() => ({{alerts: []}})),
      fetch('/golf_3ball').then(r => r.json()).catch(() => ({{edges: []}})),
    ]);
    updateDials(stats); renderKpis(stats);
    buildCurveChart(stats.pnl_curve || []);
    buildCalibChart(stats.edge_calibration || []);
    renderBucketTable('tbody-sport',           stats.by_sport);
    renderBucketTable('tbody-favdog',          stats.by_fav_dog);
    renderBucketTable('tbody-side',            stats.by_our_side);
    renderBucketTable('tbody-provider',        stats.by_provider);
    renderBucketTable('tbody-entry',           stats.by_entry_bucket);
    renderBucketTable('tbody-tip',             stats.by_tip_bucket);
    renderBucketTable('tbody-exit',            stats.by_exit_reason);
    renderBucketTable('tbody-series',          stats.by_series);
    renderBucketTable('tbody-hold',            stats.by_hold);
    renderBucketTable('tbody-confidence',      stats.by_confidence);
    renderBucketTable('tbody-exit-policy',     stats.by_exit_policy);
    renderBucketTable('tbody-edge-bucket',     stats.by_edge_bucket);
    renderBucketTable('tbody-whale-class',     stats.by_whale_class);
    renderBucketTable('tbody-whale-magnitude', stats.by_whale_magnitude);
    renderBucketTable('tbody-whale-side',      stats.by_whale_side);
    renderBucketTable('tbody-side-x-entry',    stats.by_side_x_entry);
    renderClvTable('tbody-clv-sport',          stats.by_sport_clv);
    renderWindowedPnl(stats.windowed_pnl, stats.open_mtm);
    renderTennis(stats.tennis_summary);
    renderBacktest(stats.settlement);
    renderCrossExchange(crossEx);
    renderGolfLeader(golfLeader);
    renderGolf3ball(golf3ball);
    renderOpen(openPos); renderTrades(tradesRows); renderDaily(daily);
    document.getElementById('refresh-status').textContent = 'updated ' + new Date().toLocaleTimeString();
  }} catch (e) {{
    document.getElementById('refresh-status').textContent = 'fetch error: ' + e.message;
  }}
}}
// Phone / Laptop view toggle. Class-based so the mobile layout can be
// previewed from a desktop browser; persisted in localStorage.
const VIEW_KEY = 'edgemonitor_view';
function applyView(mode) {{
  const shell = document.querySelector('.shell');
  const btn = document.getElementById('view-toggle');
  if (mode === 'phone') {{
    shell.classList.add('phone-view');
    btn.textContent = '📱 Phone';
  }} else {{
    shell.classList.remove('phone-view');
    btn.textContent = '💻 Laptop';
  }}
}}
function toggleView() {{
  const cur = (localStorage.getItem(VIEW_KEY) === 'phone') ? 'phone' : 'laptop';
  const next = (cur === 'phone') ? 'laptop' : 'phone';
  localStorage.setItem(VIEW_KEY, next);
  applyView(next);
}}
applyView(localStorage.getItem(VIEW_KEY) === 'phone' ? 'phone' : 'laptop');

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""


def main() -> None:
    env = env_config()
    port = int(os.environ.get("PORT", env.dashboard_port))
    uvicorn.run("src.dashboard:app", host=env.dashboard_host, port=port, reload=False)


if __name__ == "__main__":
    main()
