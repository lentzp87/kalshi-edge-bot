"""Soccer match model.

Soccer is a 3-way market underneath (home / away / draw), but Kalshi's
"Will TEAM win?" markets are 2-way (yes / no). The "no" side absorbs
both "draw" and "opposite team wins". We:

  1. Fetch 3-way odds from The Odds API (covers EPL, MLS, Champions
     League, La Liga, etc.).
  2. Devig across all three outcomes so probabilities sum to 1.
  3. Return p_home_wins to the caller — Kalshi's YES side maps directly.
  4. Implicit p_no = 1 - p_home_wins (includes draw probability).

Kalshi soccer series (best guesses — bot logs unknown):
    KXSOCCERMATCH-...
    KXEPLGAME-...
    KXMLSGAME-...
    KXCHAMPSGAME-...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from ..config import file_config
from ..kalshi_client import Market
from ..market_fields import (
    event_start_utc, name_match,
    teams_from_rules, teams_from_title, yes_side_name,
)
from ..odds_provider import fair_probability_for_soccer
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefixes we handle. Add more as the bot logs unknown ones.
SOCCER_SERIES: set[str] = {
    "KXSOCCERMATCH", "KXEPLGAME", "KXMLSGAME", "KXCHAMPSGAME",
    "KXLALIGAGAME", "KXSERIE", "KXBUNDESLIGAGAME",
}

# Trade window (soccer matches are typically scheduled days in advance)
SOCCER_MAX_MIN = 36 * 60
SOCCER_MIN_MIN = 5


@dataclass
class SoccerModel:
    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not file_config().models.sports.enabled:
            return None

        ticker = market.ticker or ""
        title = market.raw.get("title") or ""
        rules = market.raw.get("rules_primary") or ""

        teams = teams_from_title(title) or teams_from_rules(rules)
        if not teams:
            log.info("soccer.skip.no_team_names", ticker=ticker, title=title[:120])
            return None
        # Convention is "Away vs Home" same as MLB. Soccer often shows
        # "Home v Away" but our parser handles both — we treat first
        # as away-equivalent for matching.
        team_a, team_b = teams

        our_name = yes_side_name(market.raw)
        if not our_name:
            log.info("soccer.skip.no_yes_side", ticker=ticker)
            return None

        tip_utc = event_start_utc(market.raw)
        date_utc_str = None
        mins_to_tip = None
        if tip_utc:
            mins_to_tip = (tip_utc - datetime.now(timezone.utc)).total_seconds() / 60
            if mins_to_tip < SOCCER_MIN_MIN:
                log.info("soccer.skip.too_close", ticker=ticker, mins=round(mins_to_tip, 1))
                return None
            if mins_to_tip > SOCCER_MAX_MIN:
                log.info("soccer.skip.too_far", ticker=ticker, mins=round(mins_to_tip, 1))
                return None
            date_utc_str = tip_utc.strftime("%Y-%m-%d")

        # The book uses home/away — we don't know which Kalshi team is
        # which, so try both orientations.
        fair = await fair_probability_for_soccer(
            home_name=team_b, away_name=team_a, date_utc=date_utc_str,
        )
        if not fair:
            fair = await fair_probability_for_soccer(
                home_name=team_a, away_name=team_b, date_utc=date_utc_str,
            )
        if not fair:
            log.info("soccer.skip.no_book_match",
                     ticker=ticker, team_a=team_a, team_b=team_b)
            return None
        p_home, p_away, provider, home_full, away_full = fair

        # Map to OUR yes side: which book team did we back?
        if name_match(our_name, home_full) or name_match(our_name, team_b):
            p_yes = p_home
            our_label = home_full
        elif name_match(our_name, away_full) or name_match(our_name, team_a):
            p_yes = p_away
            our_label = away_full
        else:
            log.info("soccer.skip.our_team_unmatched",
                     ticker=ticker, our=our_name,
                     home=home_full, away=away_full)
            return None

        p_yes = max(0.02, min(0.98, p_yes))
        # Soccer p_yes < 0.5 is common because of draws; that's fine.
        confidence = 0.70
        tip_str = f"start in {mins_to_tip:.0f}min" if mins_to_tip is not None else "no start"
        reason = (
            f"SOCCER pregame {away_full}@{home_full} | "
            f"book[{provider}] p_home={p_home:.3f} p_away={p_away:.3f} "
            f"(p_draw={1 - p_home - p_away:.3f}) | "
            f"our={our_label} p_yes={p_yes:.3f} | {tip_str}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
