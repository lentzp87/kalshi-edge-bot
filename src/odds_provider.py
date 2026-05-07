"""Sportsbook odds provider.

Two-tier strategy:

  Tier 1 (preferred): The Odds API (https://the-odds-api.com)
    - Multi-book consensus (typically 5-15 books per game)
    - Free tier: 500 req/month, paid tiers from $30/mo for 100k req/mo
    - Set ODDS_API_KEY env var to enable
    - Far better than single-book — averages out sportsbook-specific lean

  Tier 2 (fallback): ESPN pickcenter
    - Single book (typically DraftKings)
    - Free, already in our ESPN summary fetch
    - Better than nothing; weaker than multi-book

The model calls `fair_probability_for_game(...)` and we transparently
pick the best available source. Provider name is returned alongside so
the journal can record which source was used.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
import structlog

from .devig import devig_two_way
from .espn_client import KALSHI_TO_ESPN_ABBR, fetch_summary

log = structlog.get_logger(__name__)

# ---- The Odds API integration ----

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_ODDS_API_TTL_SEC = 90
_odds_api_cache: dict[str, tuple[float, list[dict]]] = {}
_odds_api_locks: dict[str, asyncio.Lock] = {}

# Map our internal sport key to The Odds API sport_key.
# https://the-odds-api.com/sports-odds-data/sports-apis.html
ODDS_API_SPORTS: dict[str, str] = {
    "nba":  "basketball_nba",
    "nfl":  "americanfootball_nfl",
    "mlb":  "baseball_mlb",
    "nhl":  "icehockey_nhl",
}


# SECURITY NOTE: This key is checked into source code. The repo at
# github.com/lentzp87/kalshi-edge-bot was public at one point — if it
# is still public, anyone can read this key and burn through the free
# tier. Either flip the repo to private or rotate this key after the
# trial period:
#   1. https://the-odds-api.com -> dashboard -> regenerate
#   2. gh repo edit lentzp87/kalshi-edge-bot --visibility private
# Setting the ODDS_API_KEY env var on Render takes precedence, which
# is the right way to handle this once the repo is in stable use.
_BAKED_ODDS_API_KEY = "3add2c265b411520b11f2370c4bbcf08"


def _odds_api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY") or _BAKED_ODDS_API_KEY or None


async def _fetch_odds_api(sport_key: str) -> list[dict] | None:
    """Fetch h2h (moneyline) odds across all configured books for a sport."""
    api_key = _odds_api_key()
    if not api_key or sport_key not in ODDS_API_SPORTS:
        return None

    cache_key = sport_key
    cached = _odds_api_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < _ODDS_API_TTL_SEC:
        return cached[1]

    lock = _odds_api_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _odds_api_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _ODDS_API_TTL_SEC:
            return cached[1]
        url = f"{_ODDS_API_BASE}/sports/{ODDS_API_SPORTS[sport_key]}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("odds_api.fetch_failed", err=str(e)[:120], sport=sport_key)
            return None
        _odds_api_cache[cache_key] = (time.time(), data)
        return data


def _norm_team(s: str) -> str:
    """Loose normalization to match Kalshi/ESPN abbreviations against
    The Odds API's full team names. We match on substring, lowercase.
    """
    return (s or "").upper()


async def _odds_api_fair_prob(
    *, sport: str, kalshi_away: str, kalshi_home: str, date_utc: str | None,
) -> tuple[float, float, str] | None:
    """Find the matching event in The Odds API response and return
    (fair_home, fair_away, provider) using the median across books.
    """
    events = await _fetch_odds_api(sport)
    if not events:
        return None

    # Normalize the Kalshi abbreviations through ESPN translation since the
    # Odds API uses full team names. We match on team-name substring.
    away_token = KALSHI_TO_ESPN_ABBR.get(kalshi_away.upper(), kalshi_away).upper()
    home_token = KALSHI_TO_ESPN_ABBR.get(kalshi_home.upper(), kalshi_home).upper()

    for ev in events:
        away_full = (ev.get("away_team") or "").upper()
        home_full = (ev.get("home_team") or "").upper()
        # Loose substring match — handles "Knicks" vs "NY"; we allow the
        # 2-3 letter token to appear anywhere in the full name.
        away_match = (away_token in away_full) or any(
            t in away_full for t in away_token.split()
        )
        home_match = (home_token in home_full) or any(
            t in home_full for t in home_token.split()
        )
        if not (away_match and home_match):
            continue

        # Optional date filter (±1 day to bridge ET-vs-UTC shift)
        if date_utc:
            from .espn_client import _date_set_pm1
            date_window = _date_set_pm1(date_utc)
            ev_date = (ev.get("commence_time") or "")[:10]
            if date_window and ev_date and ev_date not in date_window:
                continue

        # Aggregate de-vigged probabilities across all books, take median
        home_probs: list[float] = []
        away_probs: list[float] = []
        for book in ev.get("bookmakers") or []:
            for mkt in book.get("markets") or []:
                if mkt.get("key") != "h2h":
                    continue
                outcomes = mkt.get("outcomes") or []
                home_odds = away_odds = None
                for o in outcomes:
                    name = (o.get("name") or "").upper()
                    if name == home_full:
                        home_odds = o.get("price")
                    elif name == away_full:
                        away_odds = o.get("price")
                if home_odds is None or away_odds is None:
                    continue
                pair = devig_two_way(home_odds, away_odds)
                if pair:
                    home_probs.append(pair[0])
                    away_probs.append(pair[1])

        if not home_probs:
            continue
        # Median across books — robust to one book being weird
        home_probs.sort()
        away_probs.sort()
        n = len(home_probs)
        median_home = home_probs[n // 2]
        median_away = away_probs[n // 2]
        return median_home, median_away, f"odds_api ({n} books)"

    return None


# ---- ESPN pickcenter fallback ----

async def _espn_fair_prob(*, sport: str, espn_event_id: str) -> tuple[float, float, str] | None:
    """Pull moneyline from ESPN's pickcenter (single book, typically DraftKings)
    and return de-vigged fair probabilities (home, away, provider).
    """
    s = await fetch_summary(sport, espn_event_id)
    if not s:
        return None
    pickcenter = s.get("pickcenter") or s.get("odds") or []
    if not pickcenter:
        return None
    p = pickcenter[0]
    home_ml = (p.get("homeTeamOdds") or {}).get("moneyLine")
    away_ml = (p.get("awayTeamOdds") or {}).get("moneyLine")
    pair = devig_two_way(home_ml, away_ml)
    if not pair:
        return None
    book = (p.get("provider") or {}).get("name", "ESPN")
    return pair[0], pair[1], f"espn:{book}"


# ---- Public API used by the sports model ----

async def fair_probability_for_game(
    *, sport: str, espn_event_id: str,
    kalshi_away: str, kalshi_home: str,
    date_utc: str | None = None,
) -> tuple[float, float, str] | None:
    """Return (fair_home_prob, fair_away_prob, provider_name).

    Tries The Odds API (multi-book median) first, falls back to ESPN
    pickcenter. Returns None if no source has data.
    """
    # Tier 1: multi-book consensus (only if API key set)
    if _odds_api_key():
        r = await _odds_api_fair_prob(
            sport=sport,
            kalshi_away=kalshi_away,
            kalshi_home=kalshi_home,
            date_utc=date_utc,
        )
        if r:
            return r

    # Tier 2: ESPN pickcenter
    return await _espn_fair_prob(sport=sport, espn_event_id=espn_event_id)


# ============================================================================
# Tennis support
# ============================================================================
#
# Tennis is structured differently from team sports in The Odds API:
#   * Each TOURNAMENT has its own sport_key (e.g. tennis_atp_french_open,
#     tennis_wta_madrid_open). There can be a dozen active at once across
#     ATP, WTA, ITF, and Challenger circuits.
#   * No ESPN pickcenter fallback — tennis pickcenter coverage is spotty.
#   * Players are matched by name substring against the Kalshi player
#     token (typically 3 letters from the last name).
#
# Quota note: hitting all active tennis sport keys every loop will burn
# the free tier fast. We cache aggressively (10 min per tournament) and
# only fetch a tournament when a Kalshi market actually needs it.

_TENNIS_TTL_SEC = 600  # 10 min — tennis lines move slower than team sports
_TENNIS_SPORTS_TTL_SEC = 3600  # 1 hour — tournament list rarely changes


async def _fetch_active_tennis_sport_keys() -> list[str]:
    """Return The Odds API sport keys for currently-active tennis tournaments.

    Hits /v4/sports?all=false (active only) and filters group=="Tennis".
    Cached 1h since tournaments don't start/end frequently.
    """
    api_key = _odds_api_key()
    if not api_key:
        return []
    cache_key = "_tennis_active_sports"
    cached = _odds_api_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < _TENNIS_SPORTS_TTL_SEC:
        return cached[1]

    lock = _odds_api_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _odds_api_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _TENNIS_SPORTS_TTL_SEC:
            return cached[1]
        url = f"{_ODDS_API_BASE}/sports"
        params = {"apiKey": api_key, "all": "false"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json() or []
        except Exception as e:
            log.warning("odds_api.tennis_sports_failed", err=str(e)[:120])
            return []
        keys = [
            s["key"] for s in data
            if (s.get("group") or "").lower() == "tennis"
            and s.get("active") is not False
        ]
        _odds_api_cache[cache_key] = (time.time(), keys)
        log.info("odds_api.tennis_sports", count=len(keys), keys=keys[:8])
        return keys


async def _fetch_tennis_odds(sport_key: str) -> list[dict] | None:
    """Fetch h2h odds for a specific tennis tournament. Cached 10 min."""
    api_key = _odds_api_key()
    if not api_key:
        return None
    cache_key = f"tennis:{sport_key}"
    cached = _odds_api_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < _TENNIS_TTL_SEC:
        return cached[1]

    lock = _odds_api_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _odds_api_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _TENNIS_TTL_SEC:
            return cached[1]
        url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us,eu,uk",  # tennis books skew European
            "markets": "h2h",
            "oddsFormat": "american",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("odds_api.tennis_odds_failed",
                        err=str(e)[:120], sport_key=sport_key)
            return None
        _odds_api_cache[cache_key] = (time.time(), data)
        return data


def _player_match(token: str, full_name: str) -> bool:
    """True if a 2-4 char Kalshi player token matches a full player name.

    Tries: substring anywhere, prefix of any whitespace-separated name
    segment (handles 'TOW' -> 'Townsend' or 'Taylor Townsend').
    """
    t = (token or "").upper().strip()
    n = (full_name or "").upper().strip()
    if not t or not n:
        return False
    if t in n:
        return True
    parts = [p for p in n.replace("-", " ").replace(".", " ").split() if p]
    return any(p.startswith(t) for p in parts)


async def fair_probability_for_tennis(
    *, player_a_token: str, player_b_token: str,
    date_utc: str | None = None,
) -> tuple[float, float, str, str, str] | None:
    """Find the matching tennis match across all active tennis tournaments.

    Returns (fair_a, fair_b, provider, player_a_full, player_b_full) or
    None if no match is found.

    Player A is whichever of (home_team, away_team) matches player_a_token;
    fair_a is its de-vigged probability.
    """
    sport_keys = await _fetch_active_tennis_sport_keys()
    if not sport_keys:
        return None

    a_token = (player_a_token or "").upper()
    b_token = (player_b_token or "").upper()
    if not a_token or not b_token:
        return None

    date_window = None
    if date_utc:
        try:
            from datetime import date, timedelta
            y, m, d = (int(p) for p in date_utc.split("-"))
            d0 = date(y, m, d)
            date_window = {
                (d0 - timedelta(days=1)).isoformat(),
                d0.isoformat(),
                (d0 + timedelta(days=1)).isoformat(),
            }
        except (ValueError, AttributeError):
            date_window = None

    for sport_key in sport_keys:
        events = await _fetch_tennis_odds(sport_key)
        if not events:
            continue
        for ev in events:
            home = ev.get("home_team") or ""
            away = ev.get("away_team") or ""

            # Determine which Odds-API slot is player_a
            if _player_match(a_token, home) and _player_match(b_token, away):
                a_full, b_full = home, away
                a_is_home = True
            elif _player_match(a_token, away) and _player_match(b_token, home):
                a_full, b_full = away, home
                a_is_home = False
            else:
                continue

            # Date filter (±1 day)
            if date_window:
                ev_date = (ev.get("commence_time") or "")[:10]
                if ev_date and ev_date not in date_window:
                    continue

            # Aggregate de-vigged probs across books, take median
            home_probs: list[float] = []
            away_probs: list[float] = []
            for book in ev.get("bookmakers") or []:
                for mkt in book.get("markets") or []:
                    if mkt.get("key") != "h2h":
                        continue
                    home_odds = away_odds = None
                    for o in mkt.get("outcomes") or []:
                        name = o.get("name") or ""
                        if name == home:
                            home_odds = o.get("price")
                        elif name == away:
                            away_odds = o.get("price")
                    if home_odds is None or away_odds is None:
                        continue
                    pair = devig_two_way(home_odds, away_odds)
                    if pair:
                        home_probs.append(pair[0])
                        away_probs.append(pair[1])

            if not home_probs:
                continue
            home_probs.sort()
            away_probs.sort()
            n = len(home_probs)
            median_home = home_probs[n // 2]
            median_away = away_probs[n // 2]

            if a_is_home:
                fair_a, fair_b = median_home, median_away
            else:
                fair_a, fair_b = median_away, median_home
            provider = f"odds_api:{sport_key} ({n} books)"
            return fair_a, fair_b, provider, a_full, b_full

    return None
