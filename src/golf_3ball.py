"""Golf 3-ball / round-matchup edge advisor (beta).

READ-ONLY. This module does NOT place trades. It compares DataGolf's
single-round model probabilities against DraftKings' prices for golf
matchup markets and surfaces +EV legs for the user to bet manually.

Why this exists
---------------
DraftKings has no betting API and prohibits automated play, so the bot
can't trade there. But golf round-matchups (3-balls in Rounds 1-2,
2-balls in Rounds 3-4) are a genuinely soft market: single-round golf
is almost pure variance, which compresses every player toward 1/N, and
retail name-bias makes books shade recognizable favorites up and
obscure players down. DataGolf publishes model probabilities for these
exact matchups — their flagship product. We diff the two and ping the
user the legs where DataGolf's number beats the DK price by enough to
clear the vig.

The theory (Matt's "round matchup underdog parlay"), made disciplined:
  * The EDGE is identifying underpriced underdogs, not the parlay.
  * A parlay COMPOUNDS the vig — three 7%-hold legs ≈ 20% effective
    hold. Only parlay if every leg is independently +EV; straight bets
    always have higher EV. We flag legs; the user decides straight/parlay.

Data source
-----------
DataGolf `betting-tools/matchups` endpoint returns, per matchup, both
the `datagolf` model probabilities AND each book's implied prices
(including `draftkings`). One call gives us both sides — no DK scrape.

REQUIRES the DataGolf *betting-tools* API tier. The preds/in-play
endpoints we already use are a lower tier. If the key lacks access the
endpoint 401/403s — we log that loudly so it's obvious.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
import structlog

log = structlog.get_logger(__name__)

API_BASE = "https://feeds.datagolf.com"
CACHE_TTL_SEC = 600  # 10 min — matchup lines don't move fast

# Markets to pull. 3_balls = Rounds 1-2; round_matchups = 2-balls in
# Rounds 3-4. We try both so the advisor has content all week.
MARKETS = ("3_balls", "round_matchups")

# Minimum edge (in probability points) for a leg to be flagged. DataGolf
# percent odds are vig-free (a true model); DraftKings percent odds are
# vig-inclusive. So edge = datagolf_prob - dk_prob is already net of the
# book's hold. 0.04 = require DataGolf to like the player 4pp more than
# the DK price implies before we surface it.
MIN_EDGE_PP = 0.04

# module-level cache: (market, tour) -> (ts, payload)
_cache: dict[tuple[str, str], tuple[float, dict]] = {}
_warned_no_key = False
_warned_no_tier = False


def _api_key() -> str | None:
    return os.environ.get("DATAGOLF_API_KEY") or None


@dataclass
class MatchupEdge:
    event_name: str
    round_num: int
    market: str               # "3_balls" / "round_matchups"
    players: list[str]        # all golfers in the group
    pick: str                 # the golfer we'd back
    datagolf_prob: float      # model probability for the pick
    dk_prob: float            # DraftKings implied probability for the pick
    edge_pp: float            # datagolf_prob - dk_prob (net of vig)
    is_underdog: bool         # True if pick has the lowest DK price in the group


async def _fetch_matchups(market: str, tour: str = "pga") -> dict | None:
    """Fetch one matchup market from DataGolf betting-tools. Cached 10 min.

    Returns the parsed JSON dict, or None on missing key / tier / network.
    """
    global _warned_no_key, _warned_no_tier
    key = _api_key()
    if not key:
        if not _warned_no_key:
            log.warning("golf_3ball.no_api_key",
                        msg="DATAGOLF_API_KEY not set; 3-ball advisor disabled")
            _warned_no_key = True
        return None

    cache_key = (market, tour)
    cached = _cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < CACHE_TTL_SEC:
        return cached[1]

    url = f"{API_BASE}/betting-tools/matchups"
    params = {
        "tour": tour,
        "market": market,
        "odds_format": "percent",
        "file_format": "json",
        "key": key,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params)
            if r.status_code in (401, 403):
                if not _warned_no_tier:
                    log.warning(
                        "golf_3ball.no_tier",
                        status=r.status_code,
                        msg=("DataGolf key lacks betting-tools tier — "
                             "the matchups endpoint needs the paid betting "
                             "plan. 3-ball advisor disabled until upgraded."),
                    )
                    _warned_no_tier = True
                return None
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        log.warning("golf_3ball.fetch_failed", market=market, err=str(e)[:160])
        return None

    _cache[cache_key] = (now, data)
    log.info("golf_3ball.fetched", market=market, tour=tour,
             matchups=len(data.get("match_list", []) if isinstance(data, dict) else []))
    return data


def _player_names(m: dict) -> list[str]:
    """Pull the golfer names from a DataGolf matchup record. The endpoint
    uses p1_player_name / p2_player_name / p3_player_name keys."""
    names = []
    for slot in ("p1", "p2", "p3"):
        nm = m.get(f"{slot}_player_name")
        if nm:
            names.append(str(nm))
    return names


def _edges_from_matchup(m: dict, event_name: str, round_num: int,
                        market: str) -> list[MatchupEdge]:
    """Compute per-leg edges for one matchup. Returns one MatchupEdge per
    golfer whose DataGolf probability beats the DraftKings price.
    """
    odds = m.get("odds") or {}
    dg = odds.get("datagolf") or {}
    dk = odds.get("draftkings") or {}
    if not dg or not dk:
        return []

    names = _player_names(m)
    slots = ("p1", "p2", "p3")[:len(names)]
    if not names:
        return []

    # DK implied probabilities for the group — used to tag the underdog.
    dk_probs = {}
    for slot in slots:
        v = dk.get(slot)
        try:
            dk_probs[slot] = float(v) if v is not None else None
        except (TypeError, ValueError):
            dk_probs[slot] = None
    valid_dk = {s: p for s, p in dk_probs.items() if p is not None}
    # The underdog of the trio is the player with the LOWEST DK implied
    # probability (longest price). That's the leg Matt's theory targets.
    underdog_slot = min(valid_dk, key=valid_dk.get) if valid_dk else None

    out: list[MatchupEdge] = []
    for slot, name in zip(slots, names):
        try:
            dg_p = float(dg.get(slot)) if dg.get(slot) is not None else None
            dk_p = dk_probs.get(slot)
        except (TypeError, ValueError):
            continue
        if dg_p is None or dk_p is None or dk_p <= 0:
            continue
        edge = dg_p - dk_p
        if edge < MIN_EDGE_PP:
            continue
        out.append(MatchupEdge(
            event_name=event_name,
            round_num=round_num,
            market=market,
            players=names,
            pick=name,
            datagolf_prob=dg_p,
            dk_prob=dk_p,
            edge_pp=edge,
            is_underdog=(slot == underdog_slot),
        ))
    return out


async def find_edges(tour: str = "pga",
                     min_edge_pp: float = MIN_EDGE_PP) -> list[MatchupEdge]:
    """Scan all matchup markets and return +EV legs sorted by edge desc.

    Empty list means either: no edges today, or DataGolf betting-tools
    tier unavailable (check logs for golf_3ball.no_tier).
    """
    edges: list[MatchupEdge] = []
    for market in MARKETS:
        data = await _fetch_matchups(market, tour=tour)
        if not data or not isinstance(data, dict):
            continue
        event_name = data.get("event_name") or "?"
        try:
            round_num = int(data.get("round_num") or 0)
        except (TypeError, ValueError):
            round_num = 0
        for m in data.get("match_list", []) or []:
            if not isinstance(m, dict):
                continue
            for e in _edges_from_matchup(m, event_name, round_num, market):
                if e.edge_pp >= min_edge_pp:
                    edges.append(e)
    edges.sort(key=lambda e: e.edge_pp, reverse=True)
    return edges


def format_slack(edges: list[MatchupEdge]) -> str:
    """Format the edge list as a Slack message. Pure — doesn't post."""
    if not edges:
        return ("🏌️ *Golf matchup advisor* — no +EV legs right now "
                "(or DataGolf betting-tools tier unavailable; check logs).")
    # Group by event for readability
    by_event: dict[str, list[MatchupEdge]] = {}
    for e in edges:
        by_event.setdefault(f"{e.event_name} · R{e.round_num}", []).append(e)
    lines = ["🏌️ *Golf matchup edges* — DataGolf vs DraftKings", ""]
    for header, group in by_event.items():
        lines.append(f"*{header}*")
        for e in group:
            tag = " 🐕 underdog" if e.is_underdog else ""
            others = " / ".join(p for p in e.players if p != e.pick)
            lines.append(
                f"  ✅ *{e.pick}*{tag}  "
                f"DK {e.dk_prob*100:.0f}% · DataGolf {e.datagolf_prob*100:.0f}% "
                f"→ +{e.edge_pp*100:.1f}pp"
            )
            lines.append(f"     vs {others}")
        lines.append("")
    lines.append("_Read-only advisor. Place bets manually on DraftKings. "
                 "Parlay only if every leg clears the bar — vig compounds._")
    return "\n".join(lines)
