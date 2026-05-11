"""Cricket match model.

Kalshi cricket markets are 2-way ("Will TEAM X win?"). We:

  1. Parse the two teams out of the Kalshi market title.
  2. Fetch fair probabilities from Pinnacle (sport_id=8). Pinnacle's
     guest API has IPL (league=720), Test Matches (8896), and various
     world-cup leagues active during their seasons. We enumerate active
     leagues dynamically — each tournament is its own Pinnacle league.
  3. Map the book's two sides back to whichever side Kalshi calls YES,
     using name_match (handles "Punjab" vs "Punjab Kings", etc.).

Most cricket formats (T20, ODI, IPL) are functionally 2-way — ties are
extremely rare and resolved by super-overs. Test matches CAN draw, but
those Kalshi markets resolve "YES = team X wins" with NO covering both
"draw" and "opponent wins" — same shape as our soccer model.

Kalshi cricket series we cover (verified in the catalog 2026-05-11):
    KXIPLGAME              — Indian Premier League
    KXCRICKETT20IMATCH     — T20 international
    KXCRICKETODIMATCH      — ODI international
    KXCRICKETTESTMATCH     — Test match
    KXPSLGAME              — Pakistan Super League
    KXWPLGAME              — Women's IPL
    KXCOUNTYCHAMPMATCH     — County Championship
    KXCRICKETWOMENODIMATCH — Women's ODI
    KXCRICKETWOMENTESTMATCH — Women's Test
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
from ..odds_provider import fair_probability_for_cricket
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefixes we handle. These are the moneyline-style cricket
# series — IPL prop markets like KXIPLFIRST10 / KXIPLSIX / KXIPLFOUR
# are intentionally excluded (different model entirely).
CRICKET_SERIES: set[str] = {
    "KXIPLGAME",
    "KXCRICKETT20IMATCH",
    "KXCRICKETODIMATCH",
    "KXCRICKETTESTMATCH",
    "KXPSLGAME",
    "KXWPLGAME",
    "KXCOUNTYCHAMPMATCH",
    "KXCRICKETWOMENODIMATCH",
    "KXCRICKETWOMENTESTMATCH",
}

# Cricket schedules days ahead, lines stable for 24h+. Match length
# varies wildly: T20 = 3h, ODI = 8h, Test = up to 5 days. We trade only
# pregame so we don't have to model in-game state.
CRICKET_MAX_MIN = 36 * 60   # 36 hours
CRICKET_MIN_MIN = 5


@dataclass
class CricketModel:
    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not file_config().models.sports.enabled:
            return None

        ticker = market.ticker or ""
        title = market.raw.get("title") or ""
        rules = market.raw.get("rules_primary") or ""

        teams = teams_from_title(title) or teams_from_rules(rules)
        if not teams:
            log.info("cricket.skip.no_team_names", ticker=ticker, title=title[:120])
            return None
        team_a, team_b = teams

        our_name = yes_side_name(market.raw)
        if not our_name:
            log.info("cricket.skip.no_yes_side", ticker=ticker)
            return None

        tip_utc = event_start_utc(market.raw)
        date_utc_str = None
        mins_to_tip = None
        if tip_utc:
            mins_to_tip = (tip_utc - datetime.now(timezone.utc)).total_seconds() / 60
            if mins_to_tip < CRICKET_MIN_MIN:
                log.info("cricket.skip.too_close", ticker=ticker, mins=round(mins_to_tip, 1))
                return None
            if mins_to_tip > CRICKET_MAX_MIN:
                log.info("cricket.skip.too_far", ticker=ticker, mins=round(mins_to_tip, 1))
                return None
            date_utc_str = tip_utc.strftime("%Y-%m-%d")

        # Pinnacle's matchup orientation isn't deterministic per league
        # (some leagues call the first listed team "home"). Try both.
        fair = await fair_probability_for_cricket(
            team_a_name=team_a, team_b_name=team_b, date_utc=date_utc_str,
        )
        if not fair:
            fair = await fair_probability_for_cricket(
                team_a_name=team_b, team_b_name=team_a, date_utc=date_utc_str,
            )
        if not fair:
            log.info("cricket.skip.no_book_match",
                     ticker=ticker, team_a=team_a, team_b=team_b)
            return None
        p_a, p_b, provider, a_full, b_full = fair

        # Map to OUR yes side
        if name_match(our_name, a_full) or name_match(our_name, team_a):
            p_yes = p_a
            our_label = a_full
        elif name_match(our_name, b_full) or name_match(our_name, team_b):
            p_yes = p_b
            our_label = b_full
        else:
            log.info("cricket.skip.our_team_unmatched",
                     ticker=ticker, our=our_name,
                     a=a_full, b=b_full)
            return None

        # Pinnacle is sharp; clamp wide.
        p_yes = max(0.02, min(0.98, p_yes))
        # Slightly lower than other team sports — cricket has more
        # in-game variance (a single over can swing odds 20pp), so we
        # discount our pregame opinion a touch.
        confidence = 0.70
        tip_str = f"start in {mins_to_tip:.0f}min" if mins_to_tip is not None else "no start"
        reason = (
            f"CRICKET pregame {a_full} vs {b_full} | "
            f"book[{provider}] p_a={p_a:.3f} p_b={p_b:.3f} | "
            f"our={our_label} p_yes={p_yes:.3f} | {tip_str}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
