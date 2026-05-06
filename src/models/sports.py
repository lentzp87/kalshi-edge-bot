"""Late-game sports model.

Edge thesis
-----------
Kalshi's market price for a given team to win the game lags ESPN's
real-time win-probability model during in-game volatility. We trade
the gap whenever:

    abs(ESPN_WP_for_our_team - Kalshi_market_price) >= min_edge

We restrict to LATE GAME ONLY because that's when:
  (a) WP estimates are tight (low variance — score and time dominate)
  (b) Kalshi prices most often misprice (less time for arbs to clear)

Late-game cutoffs by sport:
  MLB: top/bottom of 7th inning or later
  NBA: 4th quarter, < 8 minutes remaining
  NHL: 3rd period
  NFL: 4th quarter

We *don't* run a WP model ourselves — ESPN's `homeWinPercentage` is
the gold-standard WP that broadcast graphics use, so we just consume it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

from ..config import file_config
from ..espn_client import find_live_game, latest_home_win_prob
from ..kalshi_client import Market
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Map a Kalshi series_ticker to (espn sport key, "late game" predicate).
def _is_mlb_late(c: dict) -> bool:
    return c["state"] == "in" and c["period"] >= 7


def _is_nba_late(c: dict) -> bool:
    if c["state"] != "in":
        return False
    if c["period"] < 4:
        return False
    secs = _clock_to_seconds(c["clock"])
    return secs is not None and secs <= 8 * 60


def _is_nhl_late(c: dict) -> bool:
    return c["state"] == "in" and c["period"] >= 3


def _is_nfl_late(c: dict) -> bool:
    return c["state"] == "in" and c["period"] >= 4


def _clock_to_seconds(clk: str) -> int | None:
    if not clk:
        return None
    s = clk.strip()
    m = re.match(r"^(\d+):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    try:
        return int(float(s))
    except ValueError:
        return None


SERIES_REGISTRY: dict[str, tuple[str, Any]] = {
    "KXMLBGAME":   ("mlb", _is_mlb_late),
    "KXNBAGAME":   ("nba", _is_nba_late),
    "KXNHLGAME":   ("nhl", _is_nhl_late),
    "KXNFLGAME":   ("nfl", _is_nfl_late),
}


@dataclass
class _ParsedTicker:
    sport: str
    series: str
    away_abbr: str
    home_abbr: str
    our_team_abbr: str
    is_late_game: Any


_TICKER_RE = re.compile(
    r"""^
        (?P<series>KX[A-Z]+GAME) -
        (?P<eventcode>[A-Z0-9]+) -
        (?P<our_team>[A-Z0-9]+)
        $""", re.VERBOSE,
)


def _split_event_code(code: str) -> tuple[str, str] | None:
    """Pull away/home team abbreviations out of the event code.

    Examples:
        '26MAY082210ATLLAD' -> ('ATL', 'LAD')
        '26MAY11DETCLE'     -> ('DET', 'CLE')

    Strategy: strip the leading date prefix (digits + 3-letter month +
    optional more digits), then try splits of the remaining team-letters.
    """
    m = re.match(r"^\d{2}[A-Z]{3}\d+(?P<teams>[A-Z]+)$", code)
    if not m:
        return None
    teams = m.group("teams")
    n = len(teams)
    # Try splits, preferring symmetric 3+3 then variants
    for left in (3, 2, 4):
        right = n - left
        if 2 <= right <= 4:
            return teams[:left], teams[left:]
    return None


def parse_ticker(market: Market) -> _ParsedTicker | None:
    event_ticker = market.raw.get("event_ticker", "") or ""
    market_ticker = market.ticker or ""
    m = _TICKER_RE.match(market_ticker)
    if not m:
        return None
    series = m.group("series")
    if series not in SERIES_REGISTRY:
        return None
    sport, late_pred = SERIES_REGISTRY[series]
    our_team = m.group("our_team")

    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    eventcode = parts[1]
    team_pair = _split_event_code(eventcode)
    if not team_pair:
        return None
    a, b = team_pair
    return _ParsedTicker(
        sport=sport, series=series,
        away_abbr=a, home_abbr=b,
        our_team_abbr=our_team,
        is_late_game=late_pred,
    )


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
            return None

        comp = await find_live_game(
            parsed.sport,
            away_abbr=parsed.away_abbr,
            home_abbr=parsed.home_abbr,
        )
        if not comp:
            log.debug("sports.skip.no_espn_game",
                      ticker=market.ticker,
                      away=parsed.away_abbr, home=parsed.home_abbr)
            return None

        if not parsed.is_late_game(comp):
            log.debug("sports.skip.not_late_game",
                      ticker=market.ticker, state=comp["state"],
                      period=comp["period"], clock=comp["clock"])
            return None

        home_wp = await latest_home_win_prob(parsed.sport, comp["id"])
        if home_wp is None:
            log.info("sports.skip.no_wp_yet", ticker=market.ticker, sport=parsed.sport)
            return None

        our_is_home = (parsed.our_team_abbr.upper() == comp["home"]["abbr"].upper())
        p_yes = home_wp if our_is_home else (1.0 - home_wp)
        p_yes = max(0.02, min(0.98, p_yes))

        confidence = self._late_game_confidence(parsed.sport, comp)

        side_str = "home" if our_is_home else "away"
        score = f"{comp['away']['abbr']} {comp['away']['score']} @ {comp['home']['abbr']} {comp['home']['score']}"
        reason = (
            f"{parsed.sport.upper()} {comp['short_detail']} {score} | "
            f"ESPN home_wp={home_wp:.3f} our={parsed.our_team_abbr}"
            f"({side_str}) -> p_yes={p_yes:.3f} conf={confidence:.2f}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)

    @staticmethod
    def _late_game_confidence(sport: str, comp: dict) -> float:
        if sport == "mlb":
            inning = comp["period"]
            if inning >= 9:
                return 0.85
            if inning == 8:
                return 0.75
            return 0.65
        if sport == "nba":
            secs = _clock_to_seconds(comp["clock"]) or 999
            if secs <= 60:
                return 0.85
            if secs <= 180:
                return 0.75
            return 0.65
        if sport == "nhl":
            secs = _clock_to_seconds(comp["clock"]) or 999
            return 0.80 if secs <= 5 * 60 else 0.65
        if sport == "nfl":
            secs = _clock_to_seconds(comp["clock"]) or 999
            return 0.80 if secs <= 5 * 60 else 0.65
        return 0.6
