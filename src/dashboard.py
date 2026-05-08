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
    return out


def _entry_price_bucket(fill_price: float | None) -> str:
    if fill_price is None:
        return "?"
    p = fill_price
    if p < 0.55: return "<55"
    if p < 0.65: return "55-65"
    if p < 0.75: return "65-75"
    if p < 0.85: return "75-85"
    return "85+"


def _tip_bucket(mins: float | None) -> str:
    if mins is None:
        return "?"
    if mins < 30: return "0-30m"
    if mins < 60: return "30-60m"
    if mins < 120: return "1-2h"
    if mins < 240: return "2-4h"
    return "4h+"


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
            "_entry_bucket": _entry_price_bucket(t.get("fill_price")),
            "_tip_bucket": _tip_bucket(meta.get("mins_to_tip")),
            "_fav_dog": (
                "favorite" if (t.get("fill_price") or 0) >= 0.5 else "underdog"
            ),
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
            "by_exit_reason": [], "by_series": [],
            "n_clv": 0, "avg_clv_bp": 0.0, "pct_positive_clv": 0.0,
            "by_sport_clv": [],
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
        # ---- CLV summary -----------------------------------------------
        # CLV per trade: clv_price - fill_price (>0 if line moved our way).
        # Aggregated across trades with a non-null clv_price.
        **_clv_summary(chronological),
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
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalshi Edge Bot</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root {{
  --bg: #0b0d10; --panel: #14171c; --panel-2: #1b2027;
  --border: #232931; --text: #e8ecef; --text-dim: #8a93a0;
  --accent: #4ade80; --danger: #f87171;
  --warn: #fbbf24; --info: #60a5fa;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; }}
a {{ color: var(--info); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

.shell {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}

.header {{ display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
.header h1 {{ font-size: 20px; margin: 0; font-weight: 600; }}
.subtitle {{ color: var(--text-dim); font-size: 13px; }}
.badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{ padding: 4px 10px; border-radius: 999px; font-size: 11px; font-weight: 500;
  letter-spacing: 0.4px; text-transform: uppercase; background: var(--panel-2); border: 1px solid var(--border); }}
.badge.paper {{ color: var(--info); border-color: rgba(96,165,250,0.3); }}
.badge.live {{ color: var(--danger); border-color: rgba(248,113,113,0.3); }}
.badge.prod {{ color: var(--warn); border-color: rgba(251,191,36,0.3); }}
.badge.demo {{ color: var(--text-dim); }}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; margin: 20px 0; }}
.kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.kpi .label {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.6px; margin-bottom: 6px; }}
.kpi .value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.kpi .sub {{ font-size: 11px; color: var(--text-dim); margin-top: 4px; font-variant-numeric: tabular-nums; }}
.kpi.pos .value {{ color: var(--accent); }}
.kpi.neg .value {{ color: var(--danger); }}

.grid {{ display: grid; gap: 16px; margin-bottom: 16px; }}
.grid.cols-2 {{ grid-template-columns: 1fr 1fr; }}
.grid.cols-3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.grid.cols-4 {{ grid-template-columns: 1fr 1fr 1fr 1fr; }}
@media (max-width: 1100px) {{ .grid.cols-3, .grid.cols-4 {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 720px) {{ .grid.cols-2, .grid.cols-3, .grid.cols-4 {{ grid-template-columns: 1fr; }} }}

.panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; }}
.panel .title {{ font-size: 12px; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.6px; margin: 0 0 10px 0; display: flex; align-items: center;
  justify-content: space-between; gap: 8px; }}
.panel .title .meta {{ text-transform: none; letter-spacing: 0; color: var(--text-dim);
  font-weight: 400; font-size: 11px; }}

.section-header {{ font-size: 13px; color: var(--text); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.6px; margin: 24px 0 8px 0;
  display: flex; align-items: center; gap: 10px; }}
.section-header::after {{ content: ''; flex: 1; height: 1px; background: var(--border); }}

.chart-wrap {{ position: relative; height: 260px; }}
.chart-wrap.tall {{ height: 320px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ font-weight: 500; color: var(--text-dim); font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.4px; }}
tr:last-child td {{ border-bottom: none; }}
td.num {{ text-align: right; }}
td.pos {{ color: var(--accent); }}
td.neg {{ color: var(--danger); }}
td.muted {{ color: var(--text-dim); }}
td.ticker {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }}
.scroll {{ max-height: 380px; overflow-y: auto; }}

.empty {{ color: var(--text-dim); text-align: center; padding: 16px 0; font-style: italic; }}

.refresh-pill {{ font-size: 11px; color: var(--text-dim); }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); margin-right: 6px; vertical-align: middle;
  box-shadow: 0 0 6px var(--accent); animation: pulse 2s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

.footer {{ color: var(--text-dim); font-size: 11px; padding: 18px 0;
  text-align: center; border-top: 1px solid var(--border); margin-top: 16px; }}
</style>
</head>
<body>
<div class="shell">

  <div class="header">
    <div>
      <h1>Kalshi Edge Bot</h1>
      <div class="subtitle">
        <span class="dot"></span><span class="refresh-pill" id="refresh-status">connecting...</span>
      </div>
    </div>
    <div class="badges">
      <span class="badge {('paper' if mode == 'paper' else 'live')}">{mode}</span>
      <span class="badge {('prod' if kalshi_env == 'prod' else 'demo')}">{kalshi_env}</span>
      <span class="badge">bankroll ${bankroll:.0f}</span>
      <span class="badge">min edge {min_edge_bp}bp</span>
      <span class="badge">min entry ${min_entry_price:.2f}</span>
    </div>
  </div>

  <div class="kpis" id="kpis">
    <div class="kpi"><div class="label">Total P&amp;L</div><div class="value" id="kpi-pnl">—</div><div class="sub" id="kpi-pnl-sub"></div></div>
    <div class="kpi"><div class="label">Trades</div><div class="value" id="kpi-trades">—</div><div class="sub" id="kpi-trades-sub"></div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="value" id="kpi-winrate">—</div><div class="sub" id="kpi-winrate-sub"></div></div>
    <div class="kpi"><div class="label">Avg Edge</div><div class="value" id="kpi-edge">—</div><div class="sub">predicted</div></div>
    <div class="kpi"><div class="label">Open Positions</div><div class="value" id="kpi-open">—</div><div class="sub">capacity 15</div></div>
    <div class="kpi"><div class="label">Best / Worst</div><div class="value" id="kpi-bestworst" style="font-size:14px">—</div><div class="sub">single-trade range</div></div>
    <div class="kpi"><div class="label">Avg CLV</div><div class="value" id="kpi-clv">—</div><div class="sub" id="kpi-clv-sub">closing-line value</div></div>
  </div>

  <div class="panel">
    <div class="title">Cumulative P&amp;L <span class="meta" id="curve-meta"></span></div>
    <div class="chart-wrap tall"><canvas id="chart-curve"></canvas></div>
  </div>

  <div class="grid cols-2" style="margin-top:16px">
    <div class="panel">
      <div class="title">Edge Calibration <span class="meta">predicted vs realized $ per trade, by edge bucket</span></div>
      <div class="chart-wrap"><canvas id="chart-calibration"></canvas></div>
    </div>
    <div class="panel">
      <div class="title">Open Positions <span class="meta" id="open-count"></span></div>
      <div class="scroll">
        <table>
          <thead><tr><th>Bet</th><th>Sport</th><th>Side</th><th class="num">Size</th><th class="num">Fill</th><th class="num">Edge</th><th>Tip</th><th>Provider</th><th>Opened</th><th class="num">Held</th></tr></thead>
          <tbody id="tbody-open"><tr><td colspan="10" class="empty">loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section-header">What works, what doesn't</div>

  <div class="grid cols-3">
    <div class="panel">
      <div class="title">By Sport</div>
      <table>
        <thead><tr><th>Sport</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-sport"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="title">Favorite vs Underdog <span class="meta">entry &gt;= 50¢ = favorite</span></div>
      <table>
        <thead><tr><th>Side</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-favdog"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="title">Home vs Away</div>
      <table>
        <thead><tr><th>Side</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-side"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="grid cols-3" style="margin-top:16px">
    <div class="panel">
      <div class="title">By Sportsbook Source <span class="meta">does multi-book help?</span></div>
      <table>
        <thead><tr><th>Provider</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-provider"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="title">By Entry Price</div>
      <table>
        <thead><tr><th>Price</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-entry"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="title">By Time-to-Tip <span class="meta">when in pregame did we enter</span></div>
      <table>
        <thead><tr><th>Window</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-tip"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="grid cols-2" style="margin-top:16px">
    <div class="panel">
      <div class="title">By Exit Reason</div>
      <table>
        <thead><tr><th>Reason</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
        <tbody id="tbody-exit"><tr><td colspan="4" class="empty">none</td></tr></tbody>
      </table>
    </div>
    <div class="panel">
      <div class="title">By Series <span class="meta">drill into specific Kalshi series</span></div>
      <div class="scroll">
        <table>
          <thead><tr><th>Series</th><th class="num">N</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
          <tbody id="tbody-series"><tr><td colspan="4" class="empty">none</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="grid cols-1" style="margin-top:16px">
    <div class="panel">
      <div class="title">CLV by Sport <span class="meta">closing-line value: did the line move our way after we entered?</span></div>
      <table>
        <thead><tr><th>Sport</th><th class="num">N</th><th class="num">Avg CLV</th><th class="num">% Positive</th></tr></thead>
        <tbody id="tbody-clv-sport"><tr><td colspan="4" class="empty">awaiting tipoff samples</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="section-header">Recent activity</div>

  <div class="panel">
    <div class="title">Recent Trades <span class="meta" id="trades-count"></span></div>
    <div class="scroll">
      <table>
        <thead><tr><th>Ticker</th><th>Sport</th><th>Side</th><th class="num">Size</th><th class="num">Edge</th><th class="num">Fill</th><th class="num">Exit</th><th class="num">CLV</th><th class="num">P&amp;L</th><th class="num">Fees</th><th>Opened</th><th class="num">Held</th><th class="num">Tip</th><th>Provider</th><th>Why</th></tr></thead>
        <tbody id="tbody-trades"><tr><td colspan="15" class="empty">loading...</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="panel" style="margin-top:16px">
    <div class="title">Daily P&amp;L</div>
    <table>
      <thead><tr><th>Date</th><th class="num">Trades</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-daily"><tr><td colspan="3" class="empty">loading...</td></tr></tbody>
    </table>
  </div>

  <div class="footer">
    Auto-refreshes every 15s &middot;
    <a href="/stats">/stats</a> &middot; <a href="/edge">/edge</a> &middot;
    <a href="/trades">/trades</a> &middot; <a href="/positions">/positions</a> &middot; <a href="/pnl">/pnl</a>
  </div>

</div>

<script>
const fmt$ = n => (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
const fmtTime = iso => {{
  if (!iso) return '';
  try {{ return new Date(iso).toLocaleString(undefined, {{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}}); }}
  catch {{ return iso; }}
}};
const cls = n => n > 0 ? 'pos' : (n < 0 ? 'neg' : 'muted');
const sportFromTicker = t => {{
  const s = (t || '').split('-')[0].toUpperCase();
  return ({{KXNBAGAME:'NBA',KXNFLGAME:'NFL',KXMLBGAME:'MLB',KXNHLGAME:'NHL',KXATPMATCH:'ATP',KXWTAMATCH:'WTA'}})[s] || s;
}};
// HH:MM in user's local TZ (drops date so the column stays narrow)
const fmtClock = iso => {{
  if (!iso) return '';
  try {{
    return new Date(iso).toLocaleTimeString(undefined, {{hour:'2-digit',minute:'2-digit'}});
  }} catch {{ return ''; }}
}};
// Compact duration between two ISO timestamps: "47m", "1h 12m", "3h"
const fmtHeld = (openIso, closeIso) => {{
  if (!openIso || !closeIso) return '';
  const ms = new Date(closeIso) - new Date(openIso);
  if (!isFinite(ms) || ms < 0) return '';
  const mins = Math.round(ms / 60000);
  if (mins < 60) return mins + 'm';
  const h = Math.floor(mins / 60), m = mins % 60;
  return m === 0 ? `${{h}}h` : `${{h}}h ${{m}}m`;
}};
// Extract bits we care about from the reason string the model writes.
// Format examples:
//   "MLB pregame Pittsburgh@SF | book[odds_api (8 books)] ... |
//    our=San Francisco(home) p_yes=0.532 | tip in 145min"
//   "TENNIS pregame Taylor Townsend vs Sara Errani | book[pinnacle] ...
//    our=Townsend p_yes=0.612 | start in 60min"
const parseReason = reason => {{
  if (!reason) return {{}};
  const out = {{}};
  let m;
  if ((m = reason.match(/book\[([^\]]+)\]/))) {{
    const provider = m[1];
    if (provider.startsWith('pinnacle')) out.provider = 'pinnacle';
    else if (provider.startsWith('odds_api')) {{
      out.provider = 'odds_api';
      const bm = provider.match(/(\d+)\s+books/);
      if (bm) out.books = bm[1];
    }} else if (provider.startsWith('espn')) out.provider = 'espn';
    else out.provider = provider;
  }}
  if ((m = reason.match(/tip in (-?\d+)min/)) || (m = reason.match(/start in (-?\d+)min/))) {{
    out.minsToTip = parseInt(m[1]);
  }}
  // Matchup — take everything between "pregame" and the first " | "
  if ((m = reason.match(/pregame\s+(.+?)\s+\|/))) {{
    out.matchup = m[1].trim();
  }}
  // Our side (team or player name)
  if ((m = reason.match(/our=([^()|]+?)(?:\(|\s+p_yes|\s*\|)/))) {{
    out.ourSide = m[1].trim();
  }}
  // Home/away tag for team sports (in parens after our=)
  if ((m = reason.match(/our=[^()|]+\((home|away)\)/))) {{
    out.ourLoc = m[1];
  }}
  // Model probability we computed for our side
  if ((m = reason.match(/p_yes=([\d.]+)/))) out.pYes = parseFloat(m[1]);
  return out;
}};
// Build a human-readable trade label like "PHI 76ers (home) vs NYK"
const fmtMatchup = (r) => {{
  const meta = parseReason(r.reason);
  if (meta.matchup && meta.ourSide) {{
    const loc = meta.ourLoc ? ` (${{meta.ourLoc}})` : '';
    return `${{meta.ourSide}}${{loc}} — ${{meta.matchup}}`;
  }}
  return '';
}};

let curveChart, calibChart;

function buildCurveChart(curve) {{
  const ctx = document.getElementById('chart-curve');
  const labels = curve.map(p => fmtTime(p.ts));
  const values = curve.map(p => p.cum);
  const tooltips = curve.map(p => p.ticker + ' ' + (p.pnl >= 0 ? '+' : '') + p.pnl.toFixed(2));
  if (curveChart) {{
    curveChart.data.labels = labels;
    curveChart.data.datasets[0].data = values;
    curveChart.data.datasets[0].tooltips = tooltips;
    curveChart.update('none'); return;
  }}
  curveChart = new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets: [{{
      label: 'Cumulative P&L', data: values, tooltips,
      borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.08)',
      borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.18, fill: true,
    }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{
          label: ctx => {{
            const tt = ctx.dataset.tooltips ? ctx.dataset.tooltips[ctx.dataIndex] : '';
            return [`Cum: ${{fmt$(ctx.parsed.y)}}`, tt].filter(Boolean);
          }}
        }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8a93a0', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#8a93a0', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
      }}
    }}
  }});
}}

function buildCalibChart(buckets) {{
  const ctx = document.getElementById('chart-calibration');
  const labels = buckets.map(b => b.bucket_pp + 'bp');
  const predicted = buckets.map(b => b.predicted);
  const realized = buckets.map(b => b.realized);
  if (calibChart) {{
    calibChart.data.labels = labels;
    calibChart.data.datasets[0].data = predicted;
    calibChart.data.datasets[1].data = realized;
    calibChart.update('none'); return;
  }}
  calibChart = new Chart(ctx, {{
    type: 'bar',
    data: {{ labels, datasets: [
      {{ label: 'Predicted $', data: predicted, backgroundColor: 'rgba(96,165,250,0.55)', borderRadius: 4 }},
      {{ label: 'Realized $',  data: realized,  backgroundColor: 'rgba(74,222,128,0.55)', borderRadius: 4 }},
    ] }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{ legend: {{ labels: {{ color: '#e8ecef', boxWidth: 12, font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8a93a0' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#8a93a0', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
      }}
    }}
  }});
}}

function renderBucketTable(tbodyId, rows) {{
  const tb = document.getElementById(tbodyId);
  if (!rows || !rows.length) {{
    tb.innerHTML = '<tr><td colspan="4" class="empty">no data yet</td></tr>'; return;
  }}
  tb.innerHTML = rows.map(r => `
    <tr>
      <td>${{r.key}}</td>
      <td class="num">${{r.n}}</td>
      <td class="num">${{(r.win_rate*100).toFixed(0)}}%</td>
      <td class="num ${{cls(r.pnl)}}">${{fmt$(r.pnl)}}</td>
    </tr>`).join('');
}}

function renderClvTable(tbodyId, rows) {{
  const tb = document.getElementById(tbodyId);
  if (!rows || !rows.length) {{
    tb.innerHTML = '<tr><td colspan="4" class="empty">awaiting tipoff samples</td></tr>'; return;
  }}
  tb.innerHTML = rows.map(r => `
    <tr>
      <td>${{r.key}}</td>
      <td class="num">${{r.n}}</td>
      <td class="num ${{cls(r.avg_clv_bp)}}">${{(r.avg_clv_bp>=0?'+':'')}}${{r.avg_clv_bp.toFixed(1)}}bp</td>
      <td class="num">${{r.pct_positive.toFixed(0)}}%</td>
    </tr>`).join('');
}}

function renderOpen(positions) {{
  document.getElementById('open-count').textContent = positions.length + ' open';
  const tb = document.getElementById('tbody-open');
  if (!positions.length) {{ tb.innerHTML = '<tr><td colspan="10" class="empty">none</td></tr>'; return; }}
  const nowIso = new Date().toISOString();
  tb.innerHTML = positions.map(p => {{
    const meta = parseReason(p.reason);
    // Bet label: "Phillies (home)" or "Townsend" — what we actually backed.
    let betLabel = meta.ourSide || p.ticker.split('-').pop() || '?';
    if (meta.ourLoc) betLabel += ` <span class="muted">(${{meta.ourLoc}})</span>`;
    if (meta.matchup) betLabel += `<br><span class="muted" style="font-size:11px">${{meta.matchup}}</span>`;
    const tipCell = (meta.minsToTip != null)
      ? (meta.minsToTip < 60 ? meta.minsToTip + 'm' : (meta.minsToTip/60).toFixed(1)+'h')
      : '<span class="muted">—</span>';
    const provCell = meta.provider
      ? (meta.books ? `${{meta.provider}} <span class="muted">(${{meta.books}})</span>` : meta.provider)
      : '<span class="muted">—</span>';
    const reasonAttr = p.reason ? ` title="${{(p.reason||'').replace(/"/g,'&quot;')}}"` : '';
    return `
    <tr${{reasonAttr}}>
      <td>${{betLabel}}</td>
      <td>${{sportFromTicker(p.ticker)}}</td>
      <td>${{p.side}}</td>
      <td class="num">$${{(p.size_usd||0).toFixed(0)}}</td>
      <td class="num">${{(p.fill_price||0).toFixed(2)}}</td>
      <td class="num">${{((p.edge||0)*100).toFixed(1)}}bp</td>
      <td class="num">${{tipCell}}</td>
      <td class="muted">${{provCell}}</td>
      <td class="muted">${{fmtClock(p.opened_ts)}}</td>
      <td class="num muted">${{fmtHeld(p.opened_ts, nowIso)}}</td>
    </tr>`;
  }}).join('');
}}

function renderTrades(rows) {{
  const tb = document.getElementById('tbody-trades');
  const closed = rows.filter(r => r.closed_ts && r.pnl_usd != null);
  if (!closed.length) {{
    document.getElementById('trades-count').textContent = '0 shown';
    tb.innerHTML = '<tr><td colspan="9" class="empty">no closed trades yet</td></tr>'; return;
  }}
  const safe = closed.filter(r => Math.abs(r.pnl_usd) <= Math.max(3 * (r.size_usd||0), 200));
  // Dedup churn: collapse exact-duplicate rows (same ticker/side/fill/exit/pnl)
  // into one row with a count. Pre-fix data had this happen ~13x per game.
  const seen = new Map();
  for (const r of safe) {{
    const key = [r.ticker, r.side, r.fill_price, r.exit_price, r.pnl_usd, r.exit_reason].join('|');
    const prev = seen.get(key);
    if (prev) {{ prev._dup += 1; }}
    else {{ seen.set(key, {{ ...r, _dup: 1 }}); }}
  }}
  const deduped = [...seen.values()];
  document.getElementById('trades-count').textContent =
    deduped.length + ' unique · ' + safe.length + ' total';
  tb.innerHTML = deduped.slice(0, 50).map(r => {{
    // CLV = (clv_price - fill_price). Positive when the line moved
    // toward our fill (good — we got a better entry than close). null
    // until the sampler runs ~5 min before tipoff.
    const clvDelta = (r.clv_price != null && r.fill_price != null)
      ? (r.clv_price - r.fill_price) : null;
    const clvCell = (clvDelta != null)
      ? `<span class="${{cls(clvDelta)}}">${{(clvDelta*100>=0?'+':'')}}${{(clvDelta*100).toFixed(1)}}bp</span>`
      : '<span class="muted">—</span>';
    const dupBadge = (r._dup > 1) ? ` <span class="muted">×${{r._dup}}</span>` : '';
    const meta = parseReason(r.reason);
    const tipCell = (meta.minsToTip != null)
      ? (meta.minsToTip < 60 ? meta.minsToTip + 'm' : (meta.minsToTip/60).toFixed(1)+'h')
      : '<span class="muted">—</span>';
    const provCell = meta.provider
      ? (meta.books ? `${{meta.provider}} <span class="muted">(${{meta.books}})</span>` : meta.provider)
      : '<span class="muted">—</span>';
    // Full reason string available on row hover for deep-dive
    const reasonAttr = r.reason ? ` title="${{(r.reason||'').replace(/"/g,'&quot;')}}"` : '';
    return `
    <tr${{reasonAttr}}>
      <td class="ticker">${{r.ticker}}${{dupBadge}}</td>
      <td>${{sportFromTicker(r.ticker)}}</td>
      <td>${{r.side}}</td>
      <td class="num">$${{(r.size_usd||0).toFixed(0)}}</td>
      <td class="num">${{((r.edge||0)*100).toFixed(1)}}bp</td>
      <td class="num">${{(r.fill_price||0).toFixed(2)}}</td>
      <td class="num">${{(r.exit_price||0).toFixed(2)}}</td>
      <td class="num">${{clvCell}}</td>
      <td class="num ${{cls(r.pnl_usd)}}">${{fmt$(r.pnl_usd)}}</td>
      <td class="num muted">$${{(r.fees_usd||0).toFixed(2)}}</td>
      <td class="muted">${{fmtClock(r.opened_ts)}}</td>
      <td class="num muted">${{fmtHeld(r.opened_ts, r.closed_ts)}}</td>
      <td class="num muted">${{tipCell}}</td>
      <td class="muted">${{provCell}}</td>
      <td class="muted">${{r.exit_reason||''}}</td>
    </tr>`;
  }}).join('');
}}

function renderDaily(dailyMap) {{
  const tb = document.getElementById('tbody-daily');
  const entries = Object.entries(dailyMap || {{}})
    .filter(([d, v]) => Math.abs(v.pnl||0) <= 1000)
    .sort((a,b) => b[0].localeCompare(a[0]));
  if (!entries.length) {{ tb.innerHTML = '<tr><td colspan="3" class="empty">no closed trades yet</td></tr>'; return; }}
  tb.innerHTML = entries.map(([d, v]) => `
    <tr><td>${{d}}</td><td class="num">${{v.n}}</td><td class="num ${{cls(v.pnl)}}">${{fmt$(v.pnl)}}</td></tr>
  `).join('');
}}

function renderKpis(s) {{
  const pnl = s.total_pnl || 0;
  const e1 = document.getElementById('kpi-pnl');
  e1.textContent = fmt$(pnl);
  e1.parentElement.classList.toggle('pos', pnl > 0);
  e1.parentElement.classList.toggle('neg', pnl < 0);
  document.getElementById('kpi-pnl-sub').textContent = (s.filtered_out > 0)
    ? `excluded ${{s.filtered_out}} corrupt trades` : 'clean trades only';
  document.getElementById('kpi-trades').textContent = s.n_trades;
  document.getElementById('kpi-trades-sub').textContent = `${{s.n_wins}}W / ${{s.n_losses}}L`;
  document.getElementById('kpi-winrate').textContent = (s.win_rate*100).toFixed(0) + '%';
  document.getElementById('kpi-winrate-sub').textContent = `avg ${{fmt$(s.avg_pnl)}} per trade`;
  document.getElementById('kpi-edge').textContent = (s.avg_edge_bp||0).toFixed(0) + 'bp';
  document.getElementById('kpi-open').textContent = s.open_positions;
  document.getElementById('kpi-bestworst').textContent = `${{fmt$(s.best_trade)}} / ${{fmt$(s.worst_trade)}}`;
  // CLV KPI: green if avg CLV > 0 (you got better entries than the close on average)
  const clvBp = s.avg_clv_bp || 0;
  const clvEl = document.getElementById('kpi-clv');
  if (s.n_clv > 0) {{
    clvEl.textContent = (clvBp >= 0 ? '+' : '') + clvBp.toFixed(1) + 'bp';
    clvEl.parentElement.classList.toggle('pos', clvBp > 0);
    clvEl.parentElement.classList.toggle('neg', clvBp < 0);
    document.getElementById('kpi-clv-sub').textContent =
      `${{s.n_clv}} samples · ${{(s.pct_positive_clv||0).toFixed(0)}}% positive`;
  }} else {{
    clvEl.textContent = '—';
    document.getElementById('kpi-clv-sub').textContent = 'awaiting tipoff samples';
  }}
  document.getElementById('curve-meta').textContent = s.pnl_curve.length
    ? `${{s.n_trades}} trades · range ${{fmt$(s.worst_trade)}} ... ${{fmt$(s.best_trade)}}`
    : 'no closed trades yet';
}}

async function refresh() {{
  try {{
    const [stats, openPos, tradesRows, daily] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/positions').then(r => r.json()),
      fetch('/trades?limit=200').then(r => r.json()),
      fetch('/pnl').then(r => r.json()),
    ]);
    renderKpis(stats);
    buildCurveChart(stats.pnl_curve);
    buildCalibChart(stats.edge_calibration);
    renderBucketTable('tbody-sport',    stats.by_sport);
    renderBucketTable('tbody-favdog',   stats.by_fav_dog);
    renderBucketTable('tbody-side',     stats.by_our_side);
    renderBucketTable('tbody-provider', stats.by_provider);
    renderBucketTable('tbody-entry',    stats.by_entry_bucket);
    renderBucketTable('tbody-tip',      stats.by_tip_bucket);
    renderBucketTable('tbody-exit',     stats.by_exit_reason);
    renderBucketTable('tbody-series',   stats.by_series);
    renderClvTable('tbody-clv-sport',   stats.by_sport_clv);
    renderOpen(openPos);
    renderTrades(tradesRows);
    renderDaily(daily);
    document.getElementById('refresh-status').textContent = 'updated ' + new Date().toLocaleTimeString();
  }} catch (e) {{
    document.getElementById('refresh-status').textContent = 'fetch error: ' + e.message;
  }}
}}

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
