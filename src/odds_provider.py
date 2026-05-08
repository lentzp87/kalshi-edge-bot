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
from .espn_client import fetch_summary
from .market_fields import name_match

log = structlog.get_logger(__name__)

# ---- The Odds API integration ----

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
# Cache TTL was 90s — burned the free tier (500 req/mo) in ~4 hours
# at a 30-second scan interval. 5 min is enough resolution for pregame
# CLV — book lines on major leagues don't move meaningfully faster.
_ODDS_API_TTL_SEC = 300
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


async def _odds_api_fair_prob(
    *, sport: str, away_name: str, home_name: str, date_utc: str | None,
) -> tuple[float, float, str] | None:
    """Find the matching event in The Odds API response and return
    (fair_home, fair_away, provider) using the median across books.

    away_name / home_name are full team names (e.g. "Pittsburgh", "San
    Francisco") taken straight from Kalshi's market title. Matched via
    market_fields.name_match against The Odds API's full home_team /
    away_team strings — robust to abbreviation differences.
    """
    events = await _fetch_odds_api(sport)
    if not events:
        return None

    for ev in events:
        away_full = ev.get("away_team") or ""
        home_full = ev.get("home_team") or ""
        # name_match is symmetric/substring + punctuation-tolerant, so
        # 'Pittsburgh' matches 'Pittsburgh Pirates' and 'Knicks' matches
        # 'New York Knicks'.
        if not (name_match(away_name, away_full) and name_match(home_name, home_full)):
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
                    n = o.get("name") or ""
                    if n == home_full:
                        home_odds = o.get("price")
                    elif n == away_full:
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
    *, sport: str, espn_event_id: str | None,
    away_name: str, home_name: str,
    date_utc: str | None = None,
) -> tuple[float, float, str] | None:
    """Return (fair_home_prob, fair_away_prob, provider_name).

    Tries The Odds API (multi-book median) first, falls back to ESPN
    pickcenter. away_name / home_name are full team names from Kalshi's
    title (e.g. "Pittsburgh", "San Francisco"). espn_event_id is
    optional — only used for the ESPN fallback.
    """
    # Tier 1: multi-book consensus (only if API key set)
    if _odds_api_key():
        r = await _odds_api_fair_prob(
            sport=sport,
            away_name=away_name,
            home_name=home_name,
            date_utc=date_utc,
        )
        if r:
            return r

    # Tier 2: ESPN pickcenter (only if we have an ESPN event id)
    if espn_event_id:
        return await _espn_fair_prob(sport=sport, espn_event_id=espn_event_id)
    return None


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

_TENNIS_TTL_SEC = 1800  # 30 min — tennis lines move slowly, tournaments often have many matches
_TENNIS_SPORTS_TTL_SEC = 21600  # 6 hours — tournament list barely changes day-to-day


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
    *, player_a_name: str, player_b_name: str,
    date_utc: str | None = None,
) -> tuple[float, float, str, str, str] | None:
    """Find a tennis match across all active tennis tournaments by full
    player names. Returns (fair_a, fair_b, provider, a_full, b_full).

    Player names are full names from Kalshi's title (e.g. "Taylor
    Townsend") and are matched via name_match against The Odds API's
    home_team / away_team fields. Player A's probability is fair_a
    regardless of which slot it falls into in the book's response.
    """
    sport_keys = await _fetch_active_tennis_sport_keys()
    if not sport_keys:
        return None

    if not player_a_name or not player_b_name:
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

            # Determine which Odds-API slot is player_a using full-name match
            if name_match(player_a_name, home) and name_match(player_b_name, away):
                a_full, b_full = home, away
                a_is_home = True
            elif name_match(player_a_name, away) and name_match(player_b_name, home):
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
                        n_str = o.get("name") or ""
                        if n_str == home:
                            home_odds = o.get("price")
                        elif n_str == away:
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
