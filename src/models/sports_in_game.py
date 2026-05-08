"""In-game late-game momentum model.

Edge thesis
-----------
ESPN's `winprobability` array updates within 2-5 seconds of major
plays (home runs, lead changes, key turnovers). Kalshi's prediction
market takes 30-90 seconds to fully reprice on those same events,
because liquidity is thinner and most takers are slower. That latency
window is real edge.

We only fire LATE in games:
    * MLB: 7th inning or later (period >= 7)
    * NBA: last 8 minutes of Q4 (period >= 4 AND clock <= 8:00)
    * NHL: 3rd period (period >= 3)
    * NFL: 4th quarter (period >= 4)

Why late only? Earlier in the game, the model has too much variance
(any score change has too long to mean-revert), and Kalshi prices are
loose enough that book-style consensus would dominate. Late-game,
the home_win_prob is sharply peaked at the truth, and any Kalshi
divergence is mostly latency.

Pipeline:
    1. Series prefix -> sport.
    2. Read team names from market.title (no ticker decoding).
    3. Find the matching ESPN game; require state == "in".
    4. Late-game predicate gate per sport.
    5. Pull latest homeWinPercentage from ESPN summary.
    6. Map to our YES side via yes_sub_title.
    7. Return p_yes with high confidence (ESPN's WP model is
       empirically calibrated).

The decision layer (src/decision.py) handles fee math + Kelly sizing.
We don't lower min_edge here — late-game lag spikes can produce 5-15pp
gross edges, well above the fee curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from ..config import file_config
from ..espn_client import find_event_by_names, latest_home_win_prob
from ..kalshi_client import Market
from ..market_fields import (
    event_start_utc, name_match, series_prefix,
    teams_from_rules, teams_from_title, yes_side_name,
)
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefix -> ESPN sport key. Same mapping as the pregame model
# but kept independent so each can evolve separately.
LATEGAME_SERIES: dict[str, str] = {
    "KXNBAGAME": "nba",
    "KXNFLGAME": "nfl",
    "KXMLBGAME": "mlb",
    "KXNHLGAME": "nhl",
}


def _is_late_game(sport: str, period: int, clock: str) -> bool:
    """True if the game has reached the high-information late-game phase."""
    if sport == "mlb":
        # 7th inning or later — bullpens are in, leads are sticky
        return period >= 7
    if sport == "nfl":
        # 4th quarter — most lead changes happen here
        return period >= 4
    if sport == "nhl":
        # 3rd period — empty-net opportunities, late goals
        return period >= 3
    if sport == "nba":
        # Last 8 minutes of Q4. Clock format from ESPN: "5:32" or "0:00".
        if period < 4:
            return False
        try:
            mins, secs = clock.split(":")
            total = int(mins) * 60 + int(secs)
            return total <= 8 * 60
        except (ValueError, AttributeError):
            return False
    return False


@dataclass
class InGameSportsModel:
    """Model that fires only on in-progress games during late-game phase."""

    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not file_config().models.sports.enabled:
            return None

        ticker = market.ticker or ""
        series = series_prefix(ticker)
        sport = LATEGAME_SERIES.get(series)
        if not sport:
            return None  # not a sport this model handles

        # ----- Read structured fields -----
        title = market.raw.get("title") or ""
        rules = market.raw.get("rules_primary") or ""
        teams = teams_from_title(title) or teams_from_rules(rules)
        if not teams:
            return None
        away_name, home_name = teams

        our_name = yes_side_name(market.raw)
        if not our_name:
            return None

        # ----- Locate the ESPN game -----
        # Game start time gives us the date for the scoreboard window.
        tip_utc = event_start_utc(market.raw)
        date_utc_str = tip_utc.strftime("%Y-%m-%d") if tip_utc else None
        comp, _raw = await find_event_by_names(
            sport, away_name=away_name, home_name=home_name,
            target_date_utc=date_utc_str,
        )
        if not comp:
            return None

        # ----- Must be in-progress + late game -----
        state = comp.get("state") or ""
        if state != "in":
            return None
        period = int(comp.get("period") or 0)
        clock = comp.get("clock") or ""
        if not _is_late_game(sport, period, clock):
            log.debug("ingame.skip.too_early",
                      ticker=ticker, sport=sport, period=period, clock=clock)
            return None

        # ----- ESPN's win probability is our truth -----
        home_wp = await latest_home_win_prob(sport, comp["id"])
        if home_wp is None:
            log.info("ingame.skip.no_winprob",
                     ticker=ticker, espn_event=comp["id"])
            return None

        # ----- Map to our YES side -----
        # Match yes_sub_title against the home/away names from the title.
        if name_match(our_name, home_name):
            our_is_home = True
        elif name_match(our_name, away_name):
            our_is_home = False
        else:
            log.info("ingame.skip.our_side_unmatched",
                     ticker=ticker, our=our_name,
                     away=away_name, home=home_name)
            return None

        p_yes = home_wp if our_is_home else (1.0 - home_wp)
        p_yes = max(0.02, min(0.98, p_yes))

        # ESPN's win-prob model is empirically calibrated against decades
        # of game outcomes — it's tighter late in games than any
        # sportsbook would be. High confidence.
        confidence = 0.85

        side = "home" if our_is_home else "away"
        reason = (
            f"{sport.upper()} late-game {away_name}@{home_name} | "
            f"book[espn:winprob] home_wp={home_wp:.3f} | "
            f"our={our_name}({side}) p_yes={p_yes:.3f} | "
            f"period={period} clock={clock}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
