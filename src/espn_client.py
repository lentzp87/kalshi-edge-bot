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


async def fetch_scoreboard(
    sport_key: str, *, date_yyyymmdd: str | None = None,
) -> dict | None:
    """Return the raw scoreboard JSON for a sport. Cached ~20s.

    If date_yyyymmdd is provided (e.g. "20260507"), fetches the
    scoreboard for that specific league-local date via ESPN's `dates`
    parameter. Otherwise falls back to the rolling "current" scoreboard,
    which only includes games near the current moment and misses
    upcoming evening games when the bot runs in the morning.
    """
    if sport_key not in SPORTS:
        return None

    cache_key = f"{sport_key}:{date_yyyymmdd or 'now'}"
    cached = _scoreboard_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < _SCOREBOARD_TTL:
        return cached[1]

    lock = _locks.setdefault(f"sb-{cache_key}", asyncio.Lock())
    async with lock:
        cached = _scoreboard_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _SCOREBOARD_TTL:
            return cached[1]
        sport, league = SPORTS[sport_key]
        url = f"{_BASE}/{sport}/{league}/scoreboard"
        if date_yyyymmdd:
            url = f"{url}?dates={date_yyyymmdd}"
        data = await _get_json(url)
        if data is not None:
            _scoreboard_cache[cache_key] = (time.time(), data)
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

def _team_view(c: dict) -> dict:
    """Extract a team's matchable name fields from a competitor dict."""
    team = c.get("team") or {}
    return {
        "abbr": team.get("abbreviation", "") or "",
        "location": team.get("location", "") or "",
        "name": team.get("name", "") or "",
        "shortDisplayName": team.get("shortDisplayName", "") or "",
        "displayName": team.get("displayName", "") or "",
        "score": int(c.get("score") or 0),
    }


def parse_competition(event: dict) -> dict | None:
    """Pull a normalized view of one ESPN scoreboard event.

    Returns:
        {
          "id": str,
          "state": "pre" | "in" | "post",
          "period": int,             # inning, quarter, period
          "clock": str,              # display clock (e.g. "5:32" or "0:00")
          "short_detail": str,       # "Top 5th" or "End 8th"
          "home": {abbr, location, name, shortDisplayName, displayName, score},
          "away": {abbr, location, name, shortDisplayName, displayName, score},
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
            "home": _team_view(home),
            "away": _team_view(away),
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


def _date_set_pm1(target_date_utc: str | None) -> set[str] | None:
    """Return {date-1, date, date+1} as YYYY-MM-DD strings.

    Kalshi tickers use ET dates; ESPN events are UTC. A 9pm ET tip is
    01:00 UTC the next day, so an exact UTC date match would miss it.
    Accept ±1 day to bridge the timezone shift.
    """
    if not target_date_utc:
        return None
    from datetime import date, timedelta
    try:
        y, m, d = (int(p) for p in target_date_utc.split("-"))
        d0 = date(y, m, d)
        return {
            (d0 - timedelta(days=1)).isoformat(),
            d0.isoformat(),
            (d0 + timedelta(days=1)).isoformat(),
        }
    except (ValueError, AttributeError):
        return None


def _yyyymmdd_window(target_date_utc: str | None) -> list[str]:
    """Return [date, date+1] in YYYYMMDD form for ESPN's `dates=` param.

    We fetch both because ESPN's league-local "slate date" sometimes
    follows the team's home timezone — e.g., a 7pm Pacific NHL game on
    May 7 ET may show under May 8's slate. Fetching both is cheap.
    Returns [] if the date is unparseable.
    """
    if not target_date_utc:
        return []
    from datetime import date, timedelta
    try:
        y, m, d = (int(p) for p in target_date_utc.split("-"))
        d0 = date(y, m, d)
        return [d0.strftime("%Y%m%d"), (d0 + timedelta(days=1)).strftime("%Y%m%d")]
    except (ValueError, AttributeError):
        return []


async def fetch_scoreboard_window(
    sport_key: str, *, target_date_utc: str | None,
) -> list[dict]:
    """Fetch ESPN scoreboard events for a date window. Returns the
    flattened list of events from {date, date+1} so callers can match
    across the ET/UTC slate boundary. Falls back to the rolling
    "current" scoreboard if no date is provided.
    """
    dates = _yyyymmdd_window(target_date_utc)
    if not dates:
        sb = await fetch_scoreboard(sport_key)
        return (sb or {}).get("events", []) or []
    out: list[dict] = []
    seen_ids: set[str] = set()
    for d in dates:
        sb = await fetch_scoreboard(sport_key, date_yyyymmdd=d)
        if not sb:
            continue
        for ev in sb.get("events", []) or []:
            ev_id = str(ev.get("id") or "")
            if ev_id and ev_id not in seen_ids:
                seen_ids.add(ev_id)
                out.append(ev)
    return out


async def find_live_game(
    sport_key: str, *, away_abbr: str, home_abbr: str,
    target_date_utc: str | None = None,
) -> dict | None:
    """Find a specific game on the slate.

    Match logic: team abbreviations (with Kalshi->ESPN translation) AND,
    if target_date_utc is provided, the game date in UTC ±1 day to
    handle the ET vs UTC shift. We fetch ESPN's scoreboard explicitly
    by date so upcoming-evening games are visible during morning scans.
    """
    events = await fetch_scoreboard_window(sport_key, target_date_utc=target_date_utc)
    if not events:
        return None
    aw = _normalize_abbr(away_abbr)
    hm = _normalize_abbr(home_abbr)
    date_window = _date_set_pm1(target_date_utc)
    for ev in events:
        c = parse_competition(ev)
        if not c:
            continue
        ca = _normalize_abbr(c["away"]["abbr"])
        ch = _normalize_abbr(c["home"]["abbr"])
        if not ((ca == aw and ch == hm) or (ca == hm and ch == aw)):
            continue
        # Optional date filter (±1 day window)
        if date_window:
            ev_date = (ev.get("date") or "")[:10]  # ISO 'YYYY-MM-DDT...'
            if ev_date and ev_date not in date_window:
                continue
        return c
    return None


async def find_event_by_names(
    sport_key: str, *, away_name: str, home_name: str,
    target_date_utc: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Find an ESPN event by full team names from a Kalshi market title.

    Matches against any of {abbreviation, location, name, shortDisplayName,
    displayName} on each ESPN competitor. Returns (parsed_competition,
    raw_event) so callers can also use the raw event for start time, etc.

    This obsoletes the abbreviation-translation path (KALSHI_TO_ESPN_ABBR)
    for any caller that has Kalshi's title-derived names — much more
    robust than chasing abbreviation mismatches.
    """
    from .market_fields import name_match
    events = await fetch_scoreboard_window(sport_key, target_date_utc=target_date_utc)
    if not events:
        return None, None
    date_window = _date_set_pm1(target_date_utc)

    def _team_matches(team_view: dict, target: str) -> bool:
        for k in ("location", "shortDisplayName", "displayName", "name", "abbr"):
            if name_match(target, team_view.get(k, "")):
                return True
        return False

    for ev in events:
        c = parse_competition(ev)
        if not c:
            continue
        h = c["home"]
        a = c["away"]
        # Match in either orientation since Kalshi's "X vs Y" is usually
        # away-vs-home but we tolerate the other order for safety.
        forward = _team_matches(a, away_name) and _team_matches(h, home_name)
        reverse = _team_matches(h, away_name) and _team_matches(a, home_name)
        if not (forward or reverse):
            continue
        if date_window:
            ev_date = (ev.get("date") or "")[:10]
            if ev_date and ev_date not in date_window:
                continue
        return c, ev
    return None, None


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
