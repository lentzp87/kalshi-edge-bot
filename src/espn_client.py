"""Tiny async wrapper around ESPN's free public scoreboard / summary APIs.

ESPN exposes scoreboard endpoints at:
    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard
    https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={id}

No auth required. Generous limits. The summary endpoint includes a
`winprobability` array — ESPN's own WP model — which is what we use
as the "true" probability vs Kalshi's market price.

We cache scoreboard responses for 20s and summary responses for 25s
since live in-game data shifts continuously.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

# Maps a friendly sport key to ESPN's URL components.
SPORTS = {
    "mlb": ("baseball", "mlb"),
    "nba": ("basketball", "nba"),
    "nhl": ("hockey", "nhl"),
    "nfl": ("football", "nfl"),
    "ncaab": ("basketball", "mens-college-basketball"),
    "ncaaf": ("football", "college-football"),
}

_BASE = "https://site.api.espn.com/apis/site/v2/sports"
_SCOREBOARD_TTL = 20  # sec
_SUMMARY_TTL = 25     # sec
_USER_AGENT = "kalshi-edge-bot/0.1"

_scoreboard_cache: dict[str, tuple[float, dict]] = {}
_summary_cache: dict[str, tuple[float, dict]] = {}
_locks: dict[str, asyncio.Lock] = {}


async def _get_json(url: str) -> dict | None:
    headers = {"User-Agent": _USER_AGENT}
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.warning("espn.fetch_failed", url=url, err=str(e)[:100])
        return None


async def fetch_scoreboard(sport_key: str) -> dict | None:
    """Return the raw scoreboard JSON for a sport. Cached ~20s."""
    if sport_key not in SPORTS:
        return None
    cached = _scoreboard_cache.get(sport_key)
    now = time.time()
    if cached and (now - cached[0]) < _SCOREBOARD_TTL:
        return cached[1]

    lock = _locks.setdefault(f"sb-{sport_key}", asyncio.Lock())
    async with lock:
        cached = _scoreboard_cache.get(sport_key)
        if cached and (time.time() - cached[0]) < _SCOREBOARD_TTL:
            return cached[1]
        sport, league = SPORTS[sport_key]
        url = f"{_BASE}/{sport}/{league}/scoreboard"
        data = await _get_json(url)
        if data is not None:
            _scoreboard_cache[sport_key] = (time.time(), data)
        return data


async def fetch_summary(sport_key: str, event_id: str) -> dict | None:
    """Return the per-event summary (includes winprobability)."""
    if sport_key not in SPORTS:
        return None
    key = f"{sport_key}:{event_id}"
    cached = _summary_cache.get(key)
    now = time.time()
    if cached and (now - cached[0]) < _SUMMARY_TTL:
        return cached[1]

    lock = _locks.setdefault(f"sm-{key}", asyncio.Lock())
    async with lock:
        cached = _summary_cache.get(key)
        if cached and (time.time() - cached[0]) < _SUMMARY_TTL:
            return cached[1]
        sport, league = SPORTS[sport_key]
        url = f"{_BASE}/{sport}/{league}/summary?event={event_id}"
        data = await _get_json(url)
        if data is not None:
            _summary_cache[key] = (time.time(), data)
        return data


# ----- Higher-level helpers --------------------------------------------------

def parse_competition(event: dict) -> dict | None:
    """Pull a normalized view of one ESPN scoreboard event.

    Returns:
        {
          "id": str,
          "state": "pre" | "in" | "post",
          "period": int,             # inning, quarter, period
          "clock": str,              # display clock (e.g. "5:32" or "0:00")
          "short_detail": str,       # "Top 5th" or "End 8th"
          "home": {"abbr": "...", "score": int},
          "away": {"abbr": "...", "score": int},
        }
        or None if event shape is unexpected.
    """
    try:
        comp = (event.get("competitions") or [{}])[0]
        status = comp.get("status") or {}
        st = status.get("type") or {}
        teams = comp.get("competitors") or []
        home = next((t for t in teams if t.get("homeAway") == "home"), {})
        away = next((t for t in teams if t.get("homeAway") == "away"), {})
        return {
            "id": str(event.get("id") or comp.get("id") or ""),
            "state": st.get("state") or "",
            "period": int(status.get("period") or 0),
            "clock": status.get("displayClock") or "",
            "short_detail": st.get("shortDetail") or "",
            "home": {
                "abbr": (home.get("team") or {}).get("abbreviation", "") or "",
                "score": int(home.get("score") or 0),
            },
            "away": {
                "abbr": (away.get("team") or {}).get("abbreviation", "") or "",
                "score": int(away.get("score") or 0),
            },
        }
    except Exception:
        log.exception("espn.parse_competition_failed", event_id=event.get("id"))
        return None


# Kalshi uses some team abbreviations that differ from ESPN. Map Kalshi -> ESPN.
# Add new entries when sports.skip.no_espn_game logs reveal a fresh mismatch.
KALSHI_TO_ESPN_ABBR: dict[str, str] = {
    # MLB
    "AZ":  "ARI",   # Arizona D-backs
    "CWS": "CHW",   # Chicago White Sox
    "ATH": "OAK",   # Athletics (Kalshi uses ATH, ESPN uses OAK)
    # NBA
    "NYK": "NY",    # NY Knicks
    "SAS": "SA",    # SA Spurs
    "GSW": "GS",    # Warriors
    "NOP": "NO",    # Pelicans
    "UTA": "UTAH",  # Utah Jazz
    # NHL
    "TBL": "TB",    # Tampa Bay Lightning
    "VGK": "VGK",   # OK as-is
    "NJD": "NJ",    # NJ Devils
    "LAK": "LA",    # LA Kings
    "SJS": "SJ",    # San Jose Sharks
}


def _normalize_abbr(abbr: str) -> str:
    a = (abbr or "").upper()
    return KALSHI_TO_ESPN_ABBR.get(a, a)


async def find_live_game(
    sport_key: str, *, away_abbr: str, home_abbr: str,
    target_date_utc: str | None = None,
) -> dict | None:
    """Find a specific game on today's scoreboard.

    Match logic: team abbreviations (with Kalshi->ESPN translation) AND,
    if target_date_utc is provided, the game date in UTC. The date filter
    prevents matching tomorrow's same-teams game when both appear on the
    rolling scoreboard window.

    target_date_utc format: 'YYYY-MM-DD'.
    """
    sb = await fetch_scoreboard(sport_key)
    if not sb:
        return None
    aw = _normalize_abbr(away_abbr)
    hm = _normalize_abbr(home_abbr)
    for ev in sb.get("events", []):
        c = parse_competition(ev)
        if not c:
            continue
        ca = _normalize_abbr(c["away"]["abbr"])
        ch = _normalize_abbr(c["home"]["abbr"])
        if not ((ca == aw and ch == hm) or (ca == hm and ch == aw)):
            continue
        # Optional date filter
        if target_date_utc:
            ev_date = (ev.get("date") or "")[:10]  # ISO 'YYYY-MM-DDT...'
            if ev_date and ev_date != target_date_utc:
                continue
        return c
    return None


async def latest_home_win_prob(sport_key: str, event_id: str) -> float | None:
    """Pull the latest home-team win probability from the per-event summary.

    ESPN's `winprobability` array is keyed by play and contains
    `homeWinPercentage` between 0 and 1. The last entry is the most recent.
    """
    s = await fetch_summary(sport_key, event_id)
    if not s:
        return None
    arr = s.get("winprobability") or []
    if not arr:
        return None
    last = arr[-1] or {}
    p = last.get("homeWinPercentage")
    try:
        p = float(p)
        if 0.0 <= p <= 1.0:
            return p
    except (TypeError, ValueError):
        pass
    return None
