"""Golf round-leader alerter (live, in-play). READ-ONLY.

The edge thesis
---------------
When a recognizable star sits near the top of a leaderboard, the book
prices them efficiently — the public piles on and the line sharpens
fast. But when a *mid- or lower-tier* golfer climbs into the lead with
holes still to play, the book's win / top-5 / top-10 odds lag: the
price is still anchored to that player's longshot pre-tournament
status. DataGolf's in-play model re-rates them in real time. The gap
between DataGolf's live probability and the DraftKings price is the
edge — and it's biggest precisely for the no-names the user wants to
catch.

We don't hand-maintain a "star list." The edge filter does the work:
a marquee leader gets priced efficiently → small gap → no alert. A
no-name leader → DK lags → big gap → alert. The mispricing itself is
the mid-tier signal.

Data
----
* DataGolf `preds/in-play` (via datagolf_client.fetch_in_play) — live
  leaderboard (current_score, current_pos, thru, today) plus live
  model probabilities (win, top_5, top_10, top_20).
* DataGolf `betting-tools/outrights` — live book odds for win / top_5
  / top_10 across 13 books incl. DraftKings.
Players are matched between the two feeds by dg_id (exact integer).

This module does NOT trade. It pings Slack so the user can check the
DraftKings price and bet manually.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx
import structlog

log = structlog.get_logger(__name__)

API_BASE = "https://feeds.datagolf.com"
OUTRIGHTS_TTL_SEC = 300  # 5 min

# Trigger window: a candidate must be within LEAD_GAP strokes of the
# tournament lead AND have played at least MIN_THRU holes (meaningfully
# into the round) but not finished (thru < 18 — "holes to play").
# LEAD_GAP=2: the first live scan (CJ Cup R2) showed the juiciest
# edges — Coody +16pp on top_10 — were on players 2 strokes back.
# Being 2 back barely dents a top-5/top-10 probability, but the book
# still prices a no-name as a longshot. gap=1 would have missed every
# hit. Tighten to 1 via the /golf_leader?lead_gap=1 param for true
# co-leaders only.
LEAD_GAP_STROKES = 2
MIN_THRU = 9

# Markets we compare DataGolf in-play prob vs DraftKings price on.
LEADER_MARKETS = ("win", "top_5", "top_10")
# Minimum edge (probability points) for a market to be flagged.
MIN_EDGE_PP = 0.05

_outrights_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_warned_no_tier = False


def _api_key() -> str | None:
    return os.environ.get("DATAGOLF_API_KEY") or None


@dataclass
class LeaderAlert:
    player_name: str
    dg_id: int
    current_pos: str
    current_score: int          # tournament score vs par (negative = under)
    strokes_back: int           # 0 = leading outright
    thru: int                   # holes played this round
    holes_left: int
    round_num: int
    # market -> (datagolf_inplay_prob, draftkings_prob, edge_pp)
    market_edges: dict = field(default_factory=dict)
    best_market: str = ""
    best_edge_pp: float = 0.0


async def _fetch_outrights(market: str, tour: str = "pga") -> dict | None:
    """Fetch one outright market (win/top_5/top_10) from DataGolf
    betting-tools. Cached 5 min. None on missing key / tier / network.
    """
    global _warned_no_tier
    key = _api_key()
    if not key:
        return None
    ck = (market, tour)
    cached = _outrights_cache.get(ck)
    now = time.time()
    if cached and (now - cached[0]) < OUTRIGHTS_TTL_SEC:
        return cached[1]
    url = f"{API_BASE}/betting-tools/outrights"
    params = {
        "tour": tour, "market": market, "odds_format": "percent",
        "file_format": "json", "key": key,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            if r.status_code in (401, 403):
                if not _warned_no_tier:
                    log.warning("golf_leader.no_tier", status=r.status_code,
                                msg="DataGolf key lacks betting-tools tier")
                    _warned_no_tier = True
                return None
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("golf_leader.outrights_failed", market=market, err=str(e)[:160])
        return None
    _outrights_cache[ck] = (now, data)
    return data


def _dk_prob_by_dgid(outrights: dict | None) -> dict[int, float]:
    """Map dg_id -> DraftKings implied probability from an outrights feed."""
    out: dict[int, float] = {}
    if not outrights:
        return out
    for row in outrights.get("odds", []) or []:
        dgid = row.get("dg_id")
        dk = row.get("draftkings")
        if dgid is None or dk is None:
            continue
        try:
            out[int(dgid)] = float(dk)
        except (TypeError, ValueError):
            continue
    return out


async def find_leader_alerts(
    tour: str = "pga",
    *,
    lead_gap: int = LEAD_GAP_STROKES,
    min_thru: int = MIN_THRU,
    min_edge_pp: float = MIN_EDGE_PP,
) -> list[LeaderAlert]:
    """Detect mid-tier golfers leading / near-leading with holes to play
    whose DataGolf in-play probability beats the DraftKings price.

    Returns alerts sorted by best edge descending. Empty list = nothing
    qualifying, or DataGolf betting-tools tier unavailable (check logs).
    """
    from .datagolf_client import fetch_in_play

    players = await fetch_in_play(tour=tour)
    if not players:
        return []

    # Tournament lead = the lowest current_score in the field.
    scores = []
    for p in players:
        s = p.get("current_score")
        if s is not None:
            try:
                scores.append(int(s))
            except (TypeError, ValueError):
                pass
    if not scores:
        return []
    lead_score = min(scores)

    # Live DraftKings odds for each market, keyed by dg_id.
    dk_by_market: dict[str, dict[int, float]] = {}
    for mkt in LEADER_MARKETS:
        dk_by_market[mkt] = _dk_prob_by_dgid(await _fetch_outrights(mkt, tour))

    alerts: list[LeaderAlert] = []
    for p in players:
        try:
            cur = int(p.get("current_score"))
            thru = int(p.get("thru") or 0)
        except (TypeError, ValueError):
            continue
        # Trigger: near the lead, meaningfully into the round, not done.
        strokes_back = cur - lead_score
        if strokes_back > lead_gap:
            continue
        if thru < min_thru or thru >= 18:
            continue

        dgid = p.get("dg_id")
        try:
            dgid = int(dgid)
        except (TypeError, ValueError):
            continue

        # Edge per market: DataGolf in-play model prob vs DK price.
        market_edges = {}
        best_mkt, best_edge = "", 0.0
        for mkt in LEADER_MARKETS:
            model_p = p.get(mkt)  # in-play feed key: 'win','top_5','top_10'
            dk_p = dk_by_market.get(mkt, {}).get(dgid)
            if model_p is None or dk_p is None:
                continue
            try:
                model_p = float(model_p)
                dk_p = float(dk_p)
            except (TypeError, ValueError):
                continue
            edge = model_p - dk_p
            market_edges[mkt] = (model_p, dk_p, edge)
            if edge > best_edge:
                best_edge, best_mkt = edge, mkt

        if best_edge < min_edge_pp:
            continue

        try:
            rnd = int(p.get("round") or 0)
        except (TypeError, ValueError):
            rnd = 0
        alerts.append(LeaderAlert(
            player_name=str(p.get("player_name") or "?"),
            dg_id=dgid,
            current_pos=str(p.get("current_pos") or "?"),
            current_score=cur,
            strokes_back=strokes_back,
            thru=thru,
            holes_left=18 - thru,
            round_num=rnd,
            market_edges=market_edges,
            best_market=best_mkt,
            best_edge_pp=best_edge,
        ))

    alerts.sort(key=lambda a: a.best_edge_pp, reverse=True)
    return alerts


def _score_str(score: int) -> str:
    if score == 0:
        return "E"
    return f"{score:+d}"


def format_slack(alerts: list[LeaderAlert]) -> str:
    """Format leader alerts as a Slack message. Pure — doesn't post."""
    if not alerts:
        return ("⛳ *Round-leader watch* — no qualifying golfers right now "
                "(no near-leader with soft DK odds, or no live round).")
    lines = ["⛳ *Round-leader alert* — mid-tier golfer near the lead, soft DK odds", ""]
    for a in alerts:
        pos = "LEADING" if a.strokes_back == 0 else f"{a.strokes_back} back"
        lines.append(
            f"🚨 *{a.player_name}* — {pos} ({a.current_pos}, "
            f"{_score_str(a.current_score)}) · R{a.round_num} thru {a.thru} "
            f"({a.holes_left} to play)"
        )
        for mkt in LEADER_MARKETS:
            if mkt not in a.market_edges:
                continue
            model_p, dk_p, edge = a.market_edges[mkt]
            flag = " ⭐" if edge >= MIN_EDGE_PP else ""
            lines.append(
                f"   {mkt:6}  DK {dk_p*100:4.1f}%  ·  DataGolf {model_p*100:4.1f}%  "
                f"→ {edge*100:+.1f}pp{flag}"
            )
        lines.append("")
    lines.append("_Read-only. Check the live DraftKings price and bet manually. "
                 "Odds move fast in-play._")
    return "\n".join(lines)
