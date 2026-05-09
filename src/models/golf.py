"""Golf outright winner model.

Golf is fundamentally different from team sports:
  * N-way market: 60-156 players, each priced 1¢-25¢
  * No "opponent" — each player is just one outcome among many
  * Books cover via per-tournament outright odds (PGA Tour event,
    each major, etc.)

The hard structural problem
---------------------------
Kalshi's fee at low prices is brutal. At 5¢, fee is ~20% of stake.
At 10¢, ~14%. The viable golf trade is mid-tournament when prices
are 0.30+ — typically R3 or R4 leaders, where the model's edge
window is wider (one bad hole can shift a 60% favorite to 30%).

We don't add a hard "round 3+" gate here because Kalshi's market
title doesn't tell us the round. Instead we let the decision layer's
min_entry_price filter (currently 0.50) do the work — it'll skip
all the early-tournament longshots automatically.

Kalshi golf series (best guesses — bot logs unknowns):
    KXMASTERS, KXUSOPEN, KXTHEOPEN, KXPGACHAMP — majors
    KXPGAEVENT-... — weekly PGA Tour event
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from ..config import file_config
from ..kalshi_client import Market
from ..market_fields import event_start_utc, yes_side_name
from ..odds_provider import fair_probability_for_golf
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefixes we handle. Add more as Kalshi adds tournaments.
GOLF_SERIES: set[str] = {
    "KXMASTERS", "KXUSOPEN", "KXTHEOPEN", "KXPGACHAMP",
    "KXPGAEVENT", "KXPGATOUR", "KXLIVGOLF",
}


@dataclass
class GolfModel:
    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not file_config().models.sports.enabled:
            return None

        ticker = market.ticker or ""
        title = market.raw.get("title") or ""

        # The "yes side" for a golf market is the player. Title format
        # varies: "Will Scottie Scheffler win the Masters?" or
        # "Scottie Scheffler — Masters Winner?". yes_sub_title is the
        # canonical source.
        our_name = yes_side_name(market.raw)
        if not our_name:
            log.info("golf.skip.no_yes_side", ticker=ticker, title=title[:120])
            return None

        # Sanity: yes_side_name might return the tournament name on
        # malformed markets. Player names are usually 2 words.
        if len(our_name.split()) > 5:
            log.info("golf.skip.bad_yes_side",
                     ticker=ticker, our=our_name)
            return None

        tip_utc = event_start_utc(market.raw)
        date_utc_str = tip_utc.strftime("%Y-%m-%d") if tip_utc else None

        fair = await fair_probability_for_golf(
            player_name=our_name, date_utc=date_utc_str,
        )
        if not fair:
            log.info("golf.skip.no_book_match", ticker=ticker, player=our_name)
            return None
        p_yes, provider, matched_full = fair

        # Clamp — players can have 1-2% fair prob; clamp to 0.005 floor
        # so the decision layer's min_entry_price still does its job.
        p_yes = max(0.005, min(0.99, p_yes))

        # Lower confidence than team sports — golf outrights have a lot
        # of variance and books can disagree by 5-10pp on contenders.
        confidence = 0.65

        when = "tip " + tip_utc.isoformat() if tip_utc else "no start"
        reason = (
            f"GOLF outright {matched_full} | "
            f"book[{provider}] p_yes={p_yes:.3f} | {when}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
