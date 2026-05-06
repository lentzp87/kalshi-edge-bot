"""Pregame sports moneyline model.

Edge thesis (pregame CLV bot)
-----------------------------
We compare Kalshi's executable YES/NO ask to the de-vigged sportsbook
consensus. We trade only when the gap is large enough to overcome the
spread + Kalshi's taker fee + a slippage buffer.

Pipeline:
    1. Parse Kalshi sports market ticker -> sport, date, away, home, side.
    2. Confirm the matching ESPN game is PREGAME (state == "pre") and
       in our trading window (5-240 min before tip).
    3. Pull moneyline from sportsbook(s) — multi-book consensus via
       The Odds API if ODDS_API_KEY is set, else ESPN pickcenter.
    4. De-vig to fair probability for each side.
    5. Run a basic injury sanity check (ESPN injuries field).
    6. Return p_yes for the side this Kalshi market represents.

Decision-layer responsibility (in src/decision.py):
    7. Compute net edge using *executable* price (not midpoint),
       subtracting fee_buffer and a slippage buffer.
    8. Apply min_net_edge threshold and Kelly-cap sizing.

This file is the model only; sizing and fee math live in decision.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from ..config import file_config
from ..espn_client import fetch_scoreboard, find_live_game, parse_competition
from ..injury_check import has_risky_injury
from ..kalshi_client import Market
from ..odds_provider import fair_probability_for_game
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Trading window in minutes-to-tipoff:
#   too early (>240 min)  -> sportsbook lines still volatile
#   too late  (<5 min)    -> live lineup chaos / liquidity drain
PREGAME_MAX_MIN = 240
PREGAME_MIN_MIN = 5


SERIES_REGISTRY: dict[str, str] = {
    # Pregame-only model. Kalshi series -> ESPN sport key.
    "KXNBAGAME":   "nba",
    "KXNFLGAME":   "nfl",
    "KXMLBGAME":   "mlb",   # supported but the spec recommends NBA/NFL first
    "KXNHLGAME":   "nhl",
}


_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


_TICKER_RE = re.compile(
    r"""^
        (?P<series>KX[A-Z]+GAME) -
        (?P<eventcode>[A-Z0-9]+) -
        (?P<our_team>[A-Z0-9]+)
        $""", re.VERBOSE,
)


@dataclass
class _ParsedTicker:
    sport: str
    series: str
    away_abbr: str
    home_abbr: str
    our_team_abbr: str
    game_date_utc: str | None = None


def _split_event_code(code: str) -> tuple[str, str, str | None] | None:
    """Pull (away, home, YYYY-MM-DD) out of the event-code segment."""
    m = re.match(
        r"^(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<rest>\d+)(?P<teams>[A-Z]+)$",
        code,
    )
    if not m:
        return None
    teams = m.group("teams")
    n = len(teams)
    pair = None
    for left in (3, 2, 4):
        right = n - left
        if 2 <= right <= 4:
            pair = (teams[:left], teams[left:])
            break
    if not pair:
        return None
    date_str = None
    try:
        yy = int(m.group("yy"))
        mon = _MONTH_NUM.get(m.group("mon"))
        rest = m.group("rest")
        dd = int(rest[:2]) if len(rest) >= 2 else int(rest)
        if mon and 1 <= dd <= 31:
            date_str = f"{2000 + yy:04d}-{mon:02d}-{dd:02d}"
    except (ValueError, IndexError):
        pass
    return pair[0], pair[1], date_str


def parse_ticker(market: Market) -> _ParsedTicker | None:
    event_ticker = market.raw.get("event_ticker", "") or ""
    market_ticker = market.ticker or ""
    m = _TICKER_RE.match(market_ticker)
    if not m:
        return None
    series = m.group("series")
    if series not in SERIES_REGISTRY:
        return None
    sport = SERIES_REGISTRY[series]
    our_team = m.group("our_team")

    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    parsed = _split_event_code(parts[1])
    if not parsed:
        return None
    a, b, game_date = parsed
    return _ParsedTicker(
        sport=sport, series=series,
        away_abbr=a, home_abbr=b,
        our_team_abbr=our_team,
        game_date_utc=game_date,
    )


def _minutes_to_tip(comp: dict, scoreboard_ev: dict | None) -> float | None:
    """Best-effort: parse the event's start time from ESPN data and
    return minutes from now (UTC) until tip. Negative if game already
    started.
    """
    iso = None
    if scoreboard_ev:
        iso = scoreboard_ev.get("date")
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        delta = (dt - datetime.now(timezone.utc)).total_seconds() / 60
        return delta
    except (ValueError, TypeError):
        return None


async def _find_event_with_raw(
    sport_key: str, *, away: str, home: str, date_utc: str | None,
) -> tuple[dict | None, dict | None]:
    """Like find_live_game but also returns the raw event dict so we
    can pull the start time. Returns (parsed_competition, raw_event).
    """
    sb = await fetch_scoreboard(sport_key)
    if not sb:
        return None, None
    from ..espn_client import _normalize_abbr  # local import to keep API clean
    aw = _normalize_abbr(away)
    hm = _normalize_abbr(home)
    for ev in sb.get("events", []):
        c = parse_competition(ev)
        if not c:
            continue
        ca = _normalize_abbr(c["away"]["abbr"])
        ch = _normalize_abbr(c["home"]["abbr"])
        if not ((ca == aw and ch == hm) or (ca == hm and ch == aw)):
            continue
        if date_utc:
            ev_date = (ev.get("date") or "")[:10]
            if ev_date and ev_date != date_utc:
                continue
        return c, ev
    return None, None


@dataclass
class SportsModel:
    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = file_config().models.sports.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None

        parsed = parse_ticker(market)
        if not parsed:
            log.info("sports.skip.parse_failed",
                     ticker=market.ticker,
                     event_ticker=market.raw.get("event_ticker", ""))
            return None

        comp, raw_ev = await _find_event_with_raw(
            parsed.sport,
            away=parsed.away_abbr,
            home=parsed.home_abbr,
            date_utc=parsed.game_date_utc,
        )
        if not comp:
            log.info("sports.skip.no_espn_game",
                     ticker=market.ticker,
                     away=parsed.away_abbr, home=parsed.home_abbr,
                     date=parsed.game_date_utc)
            return None

        # Pregame-only filter
        if comp["state"] != "pre":
            log.info("sports.skip.not_pregame",
                     ticker=market.ticker, state=comp["state"], period=comp["period"])
            return None

        mins_to_tip = _minutes_to_tip(comp, raw_ev)
        if mins_to_tip is None:
            log.info("sports.skip.no_start_time", ticker=market.ticker)
            return None
        if mins_to_tip < PREGAME_MIN_MIN:
            log.info("sports.skip.too_close_to_tip",
                     ticker=market.ticker, mins_to_tip=round(mins_to_tip, 1))
            return None
        if mins_to_tip > PREGAME_MAX_MIN:
            log.info("sports.skip.too_far_from_tip",
                     ticker=market.ticker, mins_to_tip=round(mins_to_tip, 1))
            return None

        # Injury / news sanity check (skip if any active risky-status player)
        risky, listed = await has_risky_injury(parsed.sport, comp["id"])
        if risky:
            log.info("sports.skip.injuries_listed",
                     ticker=market.ticker, count=len(listed),
                     sample=listed[:3])
            return None

        # Get fair probability from sportsbook(s)
        fair = await fair_probability_for_game(
            sport=parsed.sport,
            espn_event_id=comp["id"],
            kalshi_away=parsed.away_abbr,
            kalshi_home=parsed.home_abbr,
            date_utc=parsed.game_date_utc,
        )
        if not fair:
            log.info("sports.skip.no_sportsbook_odds",
                     ticker=market.ticker, sport=parsed.sport)
            return None
        fair_home, fair_away, provider = fair

        # Map to our side
        our_is_home = (parsed.our_team_abbr.upper() == comp["home"]["abbr"].upper())
        # Translate via ESPN abbrev map for the comparison
        from ..espn_client import _normalize_abbr
        if not our_is_home:
            our_is_home = _normalize_abbr(parsed.our_team_abbr) == _normalize_abbr(comp["home"]["abbr"])
        p_yes = fair_home if our_is_home else fair_away
        p_yes = max(0.02, min(0.98, p_yes))

        # Confidence: sportsbook consensus is reasonably tight for major
        # leagues. Cap at 0.8 so the Kelly fraction stays restrained.
        confidence = 0.75

        side = "home" if our_is_home else "away"
        score = f"{comp['away']['abbr']}@{comp['home']['abbr']}"
        reason = (
            f"{parsed.sport.upper()} pregame {score} | "
            f"book[{provider}] fair_home={fair_home:.3f} fair_away={fair_away:.3f} | "
            f"our={parsed.our_team_abbr}({side}) p_yes={p_yes:.3f} | "
            f"tip in {mins_to_tip:.0f}min"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
