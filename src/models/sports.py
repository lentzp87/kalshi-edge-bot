"""Pregame sports moneyline model (NBA, NFL, MLB, NHL).

Edge thesis (pregame CLV bot)
-----------------------------
Compare Kalshi's executable YES/NO ask to the de-vigged sportsbook
consensus. Trade only when the gap is large enough to overcome the
spread + Kalshi's taker fee + a slippage buffer.

Pipeline (post-refactor — uses Kalshi's structured fields):
    1. Series prefix (KXNBAGAME etc.) -> sport key.
    2. Read teams from market.title ("Pittsburgh vs San Francisco").
    3. Read tip time from market.occurrence_datetime (UTC ISO).
    4. Read our YES side from market.yes_sub_title.
    5. Pregame window check via mins_to_tip (5-240 min).
    6. Pull moneyline from sportsbook(s):
         Tier 1: The Odds API (multi-book consensus, full-name match)
         Tier 2: ESPN pickcenter (only if Tier 1 has no data)
    7. (NBA/NFL only) Injury sanity check via ESPN summary.
    8. Return p_yes for our side.

Decision-layer responsibility (in src/decision.py):
    9. Compute net edge using *executable* price (not midpoint),
       subtracting fee_buffer and a slippage buffer.
   10. Apply min_net_edge threshold and Kelly-cap sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from ..config import file_config
from ..espn_client import find_event_by_names
from ..injury_check import has_risky_injury
from ..kalshi_client import Market
from ..market_fields import (
    event_start_utc, name_match, series_prefix,
    teams_from_rules, teams_from_title, yes_side_name,
)
from ..odds_provider import fair_probability_for_game
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Trading window in minutes-to-tipoff:
#   too late  (<120 min)   -> dashboard data shows 1-2h window has
#                              25% win rate and -$355 P&L on 28 trades.
#                              Late-pregame news (lineup scratches,
#                              weather, sharp volume) moves the book
#                              faster than our cached probabilities.
#   too early (>1440 min)  -> book lines may shift before close.
# 2-24h is the empirically-best window: 81% wr / +$96 in 2-4h, 100% in 4h+.
PREGAME_MAX_MIN = 24 * 60  # 24 hours
PREGAME_MIN_MIN = 120      # 2 hours (was 5)


SERIES_REGISTRY: dict[str, str] = {
    # Pregame-only model. Kalshi series -> ESPN sport key.
    "KXNBAGAME":    "nba",
    "KXNFLGAME":    "nfl",
    "KXMLBGAME":    "mlb",
    "KXNHLGAME":    "nhl",
    # WNBA — same template, different ESPN sport key
    "KXWNBAGAME":   "wnba",
    # NCAA basketball / football (verified series names may differ —
    # bot will log unknown series so we can confirm)
    "KXNCAABGAME":  "ncaab",
    "KXNCAAFGAME":  "ncaaf",
}


@dataclass
class SportsModel:
    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = file_config().models.sports.enabled
        # Per-sport sub-models. Each handles a specific series-prefix
        # family. SportsModel acts as a dispatcher.
        from .tennis import TennisModel
        from .sports_in_game import InGameSportsModel
        from .ufc import UFCModel
        from .soccer import SoccerModel
        from .golf import GolfModel
        self._tennis = TennisModel(enabled=self.enabled)
        self._ufc = UFCModel(enabled=self.enabled)
        self._soccer = SoccerModel(enabled=self.enabled)
        self._golf = GolfModel(enabled=self.enabled)
        # In-game targets ESPN winprob lag for team sports.
        self._in_game = InGameSportsModel(enabled=self.enabled)

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None

        ticker = market.ticker or ""
        series = series_prefix(ticker)

        # Dispatch by series prefix to the right specialized model
        from .tennis import TENNIS_SERIES
        from .ufc import UFC_SERIES
        from .soccer import SOCCER_SERIES
        from .golf import GOLF_SERIES
        if series in TENNIS_SERIES:
            return await self._tennis.estimate(market)
        if series in UFC_SERIES:
            return await self._ufc.estimate(market)
        if series in SOCCER_SERIES:
            return await self._soccer.estimate(market)
        if series in GOLF_SERIES:
            return await self._golf.estimate(market)

        # Team sports (NBA/NFL/MLB/NHL/WNBA/NCAA): try in-game first
        # (ESPN winprob lag during late-game phase), then fall through
        # to the pregame book-consensus model below.
        ig_result = await self._in_game.estimate(market)
        if ig_result:
            return ig_result

        sport = SERIES_REGISTRY.get(series)
        if not sport:
            log.info("sports.skip.unknown_series", ticker=ticker, series=series)
            return None

        # ----- Read structured fields (no ticker decoding!) -----
        title = market.raw.get("title") or ""
        rules = market.raw.get("rules_primary") or ""
        teams = teams_from_title(title) or teams_from_rules(rules)
        if not teams:
            log.info("sports.skip.no_team_names",
                     ticker=ticker, title=title[:80])
            return None
        away_name, home_name = teams  # Kalshi convention: "Away vs Home"

        our_name = yes_side_name(market.raw)
        if not our_name:
            log.info("sports.skip.no_yes_side", ticker=ticker)
            return None

        tip_utc = event_start_utc(market.raw)
        if not tip_utc:
            log.info("sports.skip.no_start_time", ticker=ticker)
            return None
        mins_to_tip = (tip_utc - datetime.now(timezone.utc)).total_seconds() / 60

        # ----- Pregame trading window -----
        if mins_to_tip < PREGAME_MIN_MIN:
            log.info("sports.skip.too_close_to_tip",
                     ticker=ticker, mins_to_tip=round(mins_to_tip, 1))
            return None
        if mins_to_tip > PREGAME_MAX_MIN:
            log.info("sports.skip.too_far_from_tip",
                     ticker=ticker, mins_to_tip=round(mins_to_tip, 1))
            return None

        # ----- ESPN lookup (only for injury check + pickcenter fallback) -----
        date_utc_str = tip_utc.strftime("%Y-%m-%d")
        comp, _raw_ev = await find_event_by_names(
            sport, away_name=away_name, home_name=home_name,
            target_date_utc=date_utc_str,
        )
        espn_event_id = comp["id"] if comp else None

        # Injury check (NBA/NFL only — disabled internally for MLB/NHL).
        # Only runs when ESPN has the event; if not, we trust the book.
        if espn_event_id:
            risky, listed = await has_risky_injury(sport, espn_event_id)
            if risky:
                log.info("sports.skip.injuries_listed",
                         ticker=ticker, count=len(listed),
                         sample=listed[:3])
                return None

        # ----- Fair probability from book consensus -----
        fair = await fair_probability_for_game(
            sport=sport,
            espn_event_id=espn_event_id,
            away_name=away_name,
            home_name=home_name,
            date_utc=date_utc_str,
        )
        if not fair:
            log.info("sports.skip.no_sportsbook_odds",
                     ticker=ticker, sport=sport,
                     away=away_name, home=home_name)
            return None
        fair_home, fair_away, provider = fair

        # ----- Map our YES side to home/away probability -----
        # Match yes_sub_title against home/away names from the title.
        if name_match(our_name, home_name):
            our_is_home = True
        elif name_match(our_name, away_name):
            our_is_home = False
        else:
            log.info("sports.skip.our_side_unmatched",
                     ticker=ticker, our=our_name,
                     away=away_name, home=home_name)
            return None

        p_yes = fair_home if our_is_home else fair_away
        p_yes = max(0.02, min(0.98, p_yes))

        # Confidence: sportsbook consensus is reasonably tight for major
        # leagues. Cap at 0.75 so the Kelly fraction stays restrained.
        confidence = 0.75

        side = "home" if our_is_home else "away"
        reason = (
            f"{sport.upper()} pregame {away_name}@{home_name} | "
            f"book[{provider}] fair_home={fair_home:.3f} fair_away={fair_away:.3f} | "
            f"our={our_name}({side}) p_yes={p_yes:.3f} | "
            f"tip in {mins_to_tip:.0f}min"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
