"""Sportsbook odds provider.

Three-tier strategy (best to worst):

  Tier 1 (preferred): Pinnacle direct (guest API)
    - Sharp single book — Pinnacle is widely considered the truest line
      and is the standard for professional CLV measurement.
    - Free, no auth (uses public guest API key)
    - Endpoint can rate-limit or return 401 if Pinnacle changes terms

  Tier 2: The Odds API (https://the-odds-api.com)
    - Multi-book consensus (typically 5-15 books per game)
    - Free tier: 500 req/month, paid tiers from $30/mo for 100k req/mo
    - Set ODDS_API_KEY env var or use the baked-in key
    - Includes Pinnacle in most regions, plus DraftKings, FanDuel, etc.

  Tier 3 (fallback): ESPN pickcenter
    - Single book (typically DraftKings)
    - Free, already in our ESPN summary fetch
    - Better than nothing; soft book, weaker than Pinnacle / multi-book

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
    "nba":   "basketball_nba",
    "wnba":  "basketball_wnba",
    "nfl":   "americanfootball_nfl",
    "mlb":   "baseball_mlb",
    "nhl":   "icehockey_nhl",
    "ncaab": "basketball_ncaab",
    "ncaaf": "americanfootball_ncaaf",
}


# Baked-in fallback key was burned through and is now dead. Removing
# entirely so the bot's ONLY source for the Odds API key is the
# ODDS_API_KEY env var. If the env var isn't set, Tier 2 is skipped
# and the bot falls through to ESPN pickcenter (Tier 3).
_BAKED_ODDS_API_KEY: str | None = None


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


# ---- Pinnacle direct (Tier 1) ----
#
# Pinnacle's guest API is what their public website uses. The X-API-Key
# below is their published guest key — it's not a secret. Pinnacle has
# been changing terms occasionally, so we treat 401/403 as "fall through
# to Tier 2" without alarming the caller.

_PINNACLE_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
_PINNACLE_GUEST_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
_PINNACLE_TTL_SEC = 300  # 5 min cache; same as Odds API

# Pinnacle league IDs. These are stable for major leagues but verify
# via /sports/{sport_id}/leagues if a sport stops returning matchups.
PINNACLE_LEAGUE_IDS: dict[str, int] = {
    "nba":   487,
    "nfl":   889,
    "mlb":   246,
    "nhl":   1456,
    "wnba":  578,
    # NCAA IDs vary by season; safe to omit (Pinnacle returns None,
    # we fall through to Odds API which has solid NCAA coverage).
}

_pinnacle_cache: dict[str, tuple[float, dict]] = {}
_pinnacle_locks: dict[str, asyncio.Lock] = {}


async def _fetch_pinnacle_league(league_id: int) -> dict | None:
    """Fetch matchups + markets for one Pinnacle league.

    Returns a combined dict {"matchups": [...], "markets": [...]} or
    None on auth/network failure (the caller falls through to Tier 2).
    """
    cache_key = f"pinnacle:{league_id}"
    cached = _pinnacle_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < _PINNACLE_TTL_SEC:
        return cached[1]

    lock = _pinnacle_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _pinnacle_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _PINNACLE_TTL_SEC:
            return cached[1]

        headers = {
            "X-API-Key": _PINNACLE_GUEST_KEY,
            "Accept": "application/json",
        }
        m_url = f"{_PINNACLE_BASE}/leagues/{league_id}/matchups"
        k_url = f"{_PINNACLE_BASE}/leagues/{league_id}/markets/straight"
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                r_m = await client.get(m_url)
                r_k = await client.get(k_url)
                # Log status codes so we can see auth issues vs empty data
                log.info("pinnacle.fetch",
                         league_id=league_id,
                         matchups_status=r_m.status_code,
                         markets_status=r_k.status_code)
                r_m.raise_for_status()
                r_k.raise_for_status()
                matchups_json = r_m.json() or []
                markets_json = r_k.json() or []
                data = {"matchups": matchups_json, "markets": markets_json}
                log.info("pinnacle.fetched",
                         league_id=league_id,
                         matchups=len(matchups_json),
                         markets=len(markets_json))
        except Exception as e:
            log.warning("pinnacle.fetch_failed",
                        err=str(e)[:200], league_id=league_id,
                        m_url=m_url, k_url=k_url)
            return None
        _pinnacle_cache[cache_key] = (time.time(), data)
        return data


def _pinnacle_participants(matchup: dict) -> tuple[str, str] | None:
    """Pull (away_name, home_name) from a Pinnacle matchup record."""
    parts = matchup.get("participants") or []
    if len(parts) < 2:
        return None
    away = home = None
    for p in parts:
        align = (p.get("alignment") or "").lower()
        nm = p.get("name") or ""
        if align == "home":
            home = nm
        elif align == "away":
            away = nm
    if away and home:
        return away, home
    # Some Pinnacle records don't tag alignment — fall back to order
    if len(parts) >= 2 and parts[0].get("name") and parts[1].get("name"):
        return parts[0]["name"], parts[1]["name"]
    return None


async def _pinnacle_fair_prob(
    *, sport: str, away_name: str, home_name: str, date_utc: str | None,
) -> tuple[float, float, str] | None:
    """Pinnacle moneyline -> de-vigged fair (home_prob, away_prob, provider)."""
    league_id = PINNACLE_LEAGUE_IDS.get(sport)
    if not league_id:
        log.info("pinnacle.skip.no_league_id", sport=sport,
                 known_sports=list(PINNACLE_LEAGUE_IDS.keys()))
        return None
    log.debug("pinnacle.try", sport=sport, league_id=league_id,
              away=away_name, home=home_name)
    data = await _fetch_pinnacle_league(league_id)
    if not data:
        # _fetch_pinnacle_league logged the actual error; just note we missed
        log.info("pinnacle.no_data", sport=sport, league_id=league_id,
                 away=away_name, home=home_name)
        return None
    matchups = data.get("matchups") or []
    markets = data.get("markets") or []
    if not matchups:
        log.info("pinnacle.empty_matchups", sport=sport, league_id=league_id)
        return None

    # Build a matchupId -> (away_name, home_name) lookup
    mu_by_id: dict[int, tuple[str, str]] = {}
    for mu in matchups:
        mid = mu.get("id")
        if not mid:
            continue
        teams = _pinnacle_participants(mu)
        if teams:
            mu_by_id[int(mid)] = teams

    # Find the matchup whose teams match Kalshi's
    target_mu_id = None
    pin_away = pin_home = None
    for mid, (a_pin, h_pin) in mu_by_id.items():
        forward = name_match(away_name, a_pin) and name_match(home_name, h_pin)
        reverse = name_match(away_name, h_pin) and name_match(home_name, a_pin)
        if forward or reverse:
            target_mu_id = mid
            pin_away, pin_home = a_pin, h_pin
            break
    if target_mu_id is None:
        # Log a sample of available teams so we can see why match failed
        sample_teams = [t for t in list(mu_by_id.values())[:5]]
        log.info("pinnacle.no_team_match",
                 sport=sport, kalshi_away=away_name, kalshi_home=home_name,
                 pinnacle_sample=sample_teams,
                 total_matchups=len(mu_by_id))
        return None

    # Find the moneyline market for that matchup
    home_odds = away_odds = None
    for mk in markets:
        if mk.get("matchupId") != target_mu_id:
            continue
        if (mk.get("type") or "").lower() != "moneyline":
            continue
        for price in mk.get("prices") or []:
            d = price.get("designation") or ""
            label = price.get("label") or ""
            # Pinnacle uses designation home/away, sometimes label
            if d == "home" or label == pin_home:
                home_odds = price.get("price")
            elif d == "away" or label == pin_away:
                away_odds = price.get("price")
        if home_odds is not None and away_odds is not None:
            break

    if home_odds is None or away_odds is None:
        return None
    pair = devig_two_way(home_odds, away_odds)
    if not pair:
        return None
    return pair[0], pair[1], "pinnacle"


# ---- Public API used by the sports model ----

async def fair_probability_for_game(
    *, sport: str, espn_event_id: str | None,
    away_name: str, home_name: str,
    date_utc: str | None = None,
) -> tuple[float, float, str] | None:
    """Return (fair_home_prob, fair_away_prob, provider_name).

    Tries Pinnacle first (sharpest book, free), then The Odds API
    (multi-book median), then ESPN pickcenter. away_name / home_name
    are full team names from Kalshi's title.
    """
    # Tier 1: Pinnacle direct
    r = await _pinnacle_fair_prob(
        sport=sport, away_name=away_name, home_name=home_name, date_utc=date_utc,
    )
    if r:
        return r

    # Tier 2: The Odds API multi-book consensus (only if API key set)
    if _odds_api_key():
        r = await _odds_api_fair_prob(
            sport=sport,
            away_name=away_name,
            home_name=home_name,
            date_utc=date_utc,
        )
        if r:
            return r

    # Tier 3: ESPN pickcenter (only if we have an ESPN event id)
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


async def _fetch_sport_keys_in_group(group_name: str, *, ttl: int) -> list[str]:
    """Generic discovery: return all Odds API sport keys in a group
    (e.g. 'Tennis', 'Mixed Martial Arts', 'Soccer', 'Golf'). Cached
    aggressively since the active list doesn't change often.

    Hits /v4/sports?all=true so we see everything (including currently-
    inactive tournaments). The `_fetch_*_odds` callers will discover
    which ones actually have data.
    """
    api_key = _odds_api_key()
    if not api_key:
        return []
    cache_key = f"_sports_in_group:{group_name.lower()}"
    cached = _odds_api_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < ttl:
        return cached[1]

    lock = _odds_api_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _odds_api_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < ttl:
            return cached[1]
        url = f"{_ODDS_API_BASE}/sports"
        # all=true returns inactive sports too. We'll filter on `active`
        # below so we hit what's likely to have data, but still log the
        # full universe so we can see what coverage exists.
        params = {"apiKey": api_key, "all": "true"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json() or []
        except Exception as e:
            log.warning("odds_api.sports_failed",
                        group=group_name, err=str(e)[:120])
            return []
        gn = group_name.lower()
        all_in_group = [s for s in data if (s.get("group") or "").lower() == gn]
        active_keys = [s["key"] for s in all_in_group if s.get("active") is True]
        inactive_keys = [s["key"] for s in all_in_group if s.get("active") is not True]
        log.info("odds_api.sports_in_group",
                 group=group_name,
                 active=len(active_keys), active_keys=active_keys[:10],
                 inactive=len(inactive_keys), inactive_sample=inactive_keys[:5])
        _odds_api_cache[cache_key] = (time.time(), active_keys)
        return active_keys


async def _fetch_active_tennis_sport_keys() -> list[str]:
    """Tennis tournament keys currently active on The Odds API."""
    return await _fetch_sport_keys_in_group("Tennis", ttl=_TENNIS_SPORTS_TTL_SEC)


async def _fetch_active_mma_sport_keys() -> list[str]:
    """MMA / UFC sport keys."""
    return await _fetch_sport_keys_in_group("Mixed Martial Arts", ttl=_TENNIS_SPORTS_TTL_SEC)


async def _fetch_active_soccer_sport_keys() -> list[str]:
    """Soccer league keys (EPL, MLS, Champions League, etc.)."""
    return await _fetch_sport_keys_in_group("Soccer", ttl=_TENNIS_SPORTS_TTL_SEC)


async def _fetch_active_golf_sport_keys() -> list[str]:
    """Golf tournament keys (PGA Tour weekly events, majors)."""
    return await _fetch_sport_keys_in_group("Golf", ttl=_TENNIS_SPORTS_TTL_SEC)


async def _fetch_h2h_odds_by_key(sport_key: str, *, ttl: int = 1800) -> list[dict] | None:
    """Generic h2h odds fetcher for any Odds API sport_key.

    Used by tennis, MMA, soccer, golf — any sport where we discover the
    sport_key dynamically rather than hard-coding it. Cached per-key
    so each tournament burns its own quota slot.
    """
    api_key = _odds_api_key()
    if not api_key:
        return None
    cache_key = f"h2h:{sport_key}"
    cached = _odds_api_cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) < ttl:
        return cached[1]

    lock = _odds_api_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = _odds_api_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < ttl:
            return cached[1]
        url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "us,eu,uk",  # cast wide; books vary by sport
            "markets": "h2h",
            "oddsFormat": "american",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("odds_api.h2h_odds_failed",
                        err=str(e)[:120], sport_key=sport_key)
            return None
        _odds_api_cache[cache_key] = (time.time(), data)
        return data


# Back-compat alias — tennis model still calls _fetch_tennis_odds.
_fetch_tennis_odds = _fetch_h2h_odds_by_key


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


async def fair_probability_two_way_dynamic(
    *, sport_keys: list[str],
    side_a_name: str, side_b_name: str,
    date_utc: str | None = None,
    label_prefix: str = "odds_api",
) -> tuple[float, float, str, str, str] | None:
    """Two-way fair probability across a list of sport_keys.

    Used by tennis, UFC, MMA, basically any 2-way market where we
    discover sport_keys dynamically (one per tournament/event group).
    Returns (fair_a, fair_b, provider_label, a_full_name, b_full_name).

    Names from Kalshi (e.g. "Tiafoe", "Khabib") are matched via
    name_match against the book's full names (home_team / away_team).
    """
    if not sport_keys or not side_a_name or not side_b_name:
        return None

    date_window: set[str] | None = None
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
        events = await _fetch_h2h_odds_by_key(sport_key)
        if not events:
            continue
        for ev in events:
            home = ev.get("home_team") or ""
            away = ev.get("away_team") or ""
            if name_match(side_a_name, home) and name_match(side_b_name, away):
                a_full, b_full = home, away
                a_is_home = True
            elif name_match(side_a_name, away) and name_match(side_b_name, home):
                a_full, b_full = away, home
                a_is_home = False
            else:
                continue
            if date_window:
                ev_date = (ev.get("commence_time") or "")[:10]
                if ev_date and ev_date not in date_window:
                    continue
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
            mh = home_probs[n // 2]
            ma = away_probs[n // 2]
            if a_is_home:
                fair_a, fair_b = mh, ma
            else:
                fair_a, fair_b = ma, mh
            provider = f"{label_prefix}:{sport_key} ({n} books)"
            return fair_a, fair_b, provider, a_full, b_full
    return None


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


# ============================================================================
# UFC / MMA — 2-way (fighter A vs fighter B)
# ============================================================================

async def fair_probability_for_ufc(
    *, fighter_a_name: str, fighter_b_name: str,
    date_utc: str | None = None,
) -> tuple[float, float, str, str, str] | None:
    sport_keys = await _fetch_active_mma_sport_keys()
    return await fair_probability_two_way_dynamic(
        sport_keys=sport_keys,
        side_a_name=fighter_a_name, side_b_name=fighter_b_name,
        date_utc=date_utc, label_prefix="odds_api[mma]",
    )


# ============================================================================
# Soccer — Kalshi markets are 2-way (Will TEAM win?), but the underlying
# soccer market is 3-way (home / away / draw). We fetch the 3-way book
# odds, devig across all three outcomes, and return p_home_wins to the
# caller. Drawing on either side counts as "no win" for that team.
# ============================================================================

async def _devig_three_way(
    home_odds, away_odds, draw_odds,
) -> tuple[float, float, float] | None:
    """Three-way devig for soccer. Returns (p_home, p_away, p_draw)."""
    from .devig import american_to_implied
    h = american_to_implied(home_odds)
    a = american_to_implied(away_odds)
    d = american_to_implied(draw_odds)
    if h is None or a is None or d is None:
        return None
    total = h + a + d
    if total <= 0:
        return None
    return h / total, a / total, d / total


async def fair_probability_for_soccer(
    *, home_name: str, away_name: str,
    date_utc: str | None = None,
) -> tuple[float, float, str, str, str] | None:
    """Returns (p_home_wins, p_away_wins, provider, home_full, away_full).

    Note: p_home_wins + p_away_wins != 1 in soccer (because of draws).
    The Kalshi caller maps "yes for HOME" to p_home_wins and "no for HOME"
    to (1 - p_home_wins) which absorbs the draw probability.
    """
    sport_keys = await _fetch_active_soccer_sport_keys()
    if not sport_keys or not home_name or not away_name:
        return None

    date_window: set[str] | None = None
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
        events = await _fetch_h2h_odds_by_key(sport_key)
        if not events:
            continue
        for ev in events:
            home = ev.get("home_team") or ""
            away = ev.get("away_team") or ""
            if not (name_match(home_name, home) and name_match(away_name, away)):
                continue
            if date_window:
                ev_date = (ev.get("commence_time") or "")[:10]
                if ev_date and ev_date not in date_window:
                    continue
            home_probs: list[float] = []
            away_probs: list[float] = []
            for book in ev.get("bookmakers") or []:
                for mkt in book.get("markets") or []:
                    if mkt.get("key") != "h2h":
                        continue
                    home_o = away_o = draw_o = None
                    for o in mkt.get("outcomes") or []:
                        n_str = (o.get("name") or "")
                        if n_str == home:
                            home_o = o.get("price")
                        elif n_str == away:
                            away_o = o.get("price")
                        elif n_str.lower() == "draw":
                            draw_o = o.get("price")
                    if home_o is None or away_o is None or draw_o is None:
                        continue
                    triple = await _devig_three_way(home_o, away_o, draw_o)
                    if triple:
                        home_probs.append(triple[0])
                        away_probs.append(triple[1])
            if not home_probs:
                continue
            home_probs.sort()
            away_probs.sort()
            n = len(home_probs)
            mh = home_probs[n // 2]
            ma = away_probs[n // 2]
            return mh, ma, f"odds_api[soccer]:{sport_key} ({n} books)", home, away
    return None


# ============================================================================
# Golf — N-way devig for tournament outright winner markets.
# Each player has implied probability ~ 1-25%. We fetch all players in
# the tournament's h2h market and normalize so probabilities sum to 1.
# Trade only mid-tournament (R3/R4) when prices are 0.30+ to clear fees.
# ============================================================================

async def fair_probability_for_golf(
    *, player_name: str,
    date_utc: str | None = None,
) -> tuple[float, str, str] | None:
    """Returns (fair_p_yes, provider, matched_player_full_name) for one
    golf player across all currently-active tournaments.

    Caller passes player_name (e.g. "Scheffler") as the side they're
    betting YES on. We find which tournament has that player, devig the
    full field, and return the player's normalized probability.
    """
    sport_keys = await _fetch_active_golf_sport_keys()
    if not sport_keys or not player_name:
        return None
    from .devig import american_to_implied

    date_window: set[str] | None = None
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
        events = await _fetch_h2h_odds_by_key(sport_key)
        if not events:
            continue
        for ev in events:
            # Golf "events" are usually a single tournament with all
            # players as outcomes inside one market. The home/away
            # fields are typically empty or both set to the tournament
            # name; what matters is the bookmakers.markets.outcomes list.
            if date_window:
                ev_date = (ev.get("commence_time") or "")[:10]
                if ev_date and ev_date not in date_window:
                    continue
            # Aggregate across books; each book has a "h2h" market with
            # 60+ outcomes. Devig per-book by summing implied and
            # normalizing.
            per_book_player_probs: list[float] = []
            matched_full_name: str | None = None
            for book in ev.get("bookmakers") or []:
                for mkt in book.get("markets") or []:
                    if mkt.get("key") != "h2h":
                        continue
                    outcomes = mkt.get("outcomes") or []
                    implied: list[tuple[str, float]] = []
                    for o in outcomes:
                        nm = o.get("name") or ""
                        price = o.get("price")
                        ip = american_to_implied(price)
                        if ip is not None:
                            implied.append((nm, ip))
                    if not implied:
                        continue
                    total = sum(ip for _, ip in implied)
                    if total <= 0:
                        continue
                    # Find our player
                    for nm, ip in implied:
                        if name_match(player_name, nm):
                            per_book_player_probs.append(ip / total)
                            matched_full_name = matched_full_name or nm
                            break
            if not per_book_player_probs:
                continue
            # Median across books
            per_book_player_probs.sort()
            n = len(per_book_player_probs)
            median_p = per_book_player_probs[n // 2]
            return (
                median_p,
                f"odds_api[golf]:{sport_key} ({n} books)",
                matched_full_name or player_name,
            )
    return None
