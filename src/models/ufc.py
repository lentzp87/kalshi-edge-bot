"""UFC / MMA fight model.

Identical structure to tennis: 2-way market (fighter A vs fighter B),
sportsbook coverage via The Odds API's MMA group. Pinnacle is not
wired for MMA league IDs (they change per event).

Kalshi UFC tickers (best guess — bot will log unknown series):
    KXUFCFIGHT-26MAY10MMACVA-MMA  (UFC Fight Night)
    KXUFCMAIN-...                  (UFC PPV main card)

Title format observed on Kalshi sports markets:
    "Will Conor McGregor win the McGregor vs Khabib fight?"
We reuse the tennis-title parser since the format is the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from ..config import file_config
from ..kalshi_client import Market
from ..market_fields import (
    event_start_utc, name_match,
    teams_from_rules, teams_from_tennis_title, teams_from_title,
    yes_side_name,
)
from ..odds_provider import fair_probability_for_ufc
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefixes we handle. Add more as the bot logs unknown ones.
UFC_SERIES: set[str] = {"KXUFCFIGHT", "KXUFCMAIN", "KXMMAFIGHT"}

# Trade window in minutes-to-fight-time
UFC_MAX_MIN = 36 * 60   # 36 hours
UFC_MIN_MIN = 5


@dataclass
class UFCModel:
    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not file_config().models.sports.enabled:
            return None

        ticker = market.ticker or ""
        title = market.raw.get("title") or ""
        rules = market.raw.get("rules_primary") or ""

        # Try tennis-style title first (sentence shape), then plain
        teams = teams_from_tennis_title(title) or teams_from_title(title) or teams_from_rules(rules)
        if not teams:
            log.info("ufc.skip.no_fighter_names", ticker=ticker, title=title[:120])
            return None
        fighter_a, fighter_b = teams

        our_name = yes_side_name(market.raw)
        if not our_name:
            log.info("ufc.skip.no_yes_side", ticker=ticker)
            return None

        tip_utc = event_start_utc(market.raw)
        date_utc_str = None
        if tip_utc:
            mins_to_tip = (tip_utc - datetime.now(timezone.utc)).total_seconds() / 60
            if mins_to_tip < UFC_MIN_MIN:
                log.info("ufc.skip.too_close", ticker=ticker, mins=round(mins_to_tip, 1))
                return None
            if mins_to_tip > UFC_MAX_MIN:
                log.info("ufc.skip.too_far", ticker=ticker, mins=round(mins_to_tip, 1))
                return None
            date_utc_str = tip_utc.strftime("%Y-%m-%d")
        else:
            mins_to_tip = None

        fair = await fair_probability_for_ufc(
            fighter_a_name=fighter_a, fighter_b_name=fighter_b,
            date_utc=date_utc_str,
        )
        if not fair:
            log.info("ufc.skip.no_book_match",
                     ticker=ticker, a=fighter_a, b=fighter_b)
            return None
        fair_a, fair_b, provider, a_full, b_full = fair

        # Map our YES side
        if name_match(our_name, fighter_a) or name_match(our_name, a_full):
            p_yes = fair_a
        elif name_match(our_name, fighter_b) or name_match(our_name, b_full):
            p_yes = fair_b
        else:
            log.info("ufc.skip.our_fighter_unmatched",
                     ticker=ticker, our=our_name, a=a_full, b=b_full)
            return None

        p_yes = max(0.02, min(0.98, p_yes))
        confidence = 0.70  # books are tight on UFC mains, looser on prelims
        tip_str = f"start in {mins_to_tip:.0f}min" if mins_to_tip is not None else "no start"
        reason = (
            f"UFC pregame {a_full} vs {b_full} | "
            f"book[{provider}] fair_a={fair_a:.3f} fair_b={fair_b:.3f} | "
            f"our={our_name} p_yes={p_yes:.3f} | {tip_str}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
