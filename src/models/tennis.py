"""Tennis match moneyline model (WTA + ATP).

Edge thesis (same as the team-sports CLV bot)
---------------------------------------------
Compare Kalshi's executable YES/NO ask to the de-vigged sportsbook
consensus across all books that price tennis (US + EU + UK regions —
tennis books skew European, so we cast a wider net than for NBA/NFL).

Tennis differs from team sports in three ways:

  1. No ESPN coverage. Tennis pickcenter is patchy and ESPN's
     scoreboard is per-tournament, not league-wide. We rely entirely on
     The Odds API.
  2. No "home/away". Players are just player_a vs player_b.
  3. No injury check (handled by the book — if a player has withdrawn
     The Odds API simply won't list the match).

Pipeline (post-refactor — uses Kalshi's structured fields):
  1. Series prefix (KXWTAMATCH / KXATPMATCH) gates dispatch.
  2. Read player names from market.title ("Player A vs Player B").
  3. Read tip time from market.occurrence_datetime.
  4. Read our YES side from market.yes_sub_title.
  5. Look up multi-book de-vigged moneyline by player full name.
  6. Return p_yes for our side.

The decision layer (src/decision.py) handles fee math + executable
price + Kelly sizing — same as for team sports.
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
from ..odds_provider import fair_probability_for_tennis
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefixes we handle here.
TENNIS_SERIES: set[str] = {"KXWTAMATCH", "KXATPMATCH"}


# Pregame / live trading window in minutes-to-start.
# Tomorrow's tennis slate posts ~24h ahead and lines are stable for
# major events. We accept anything from 5 minutes to 36 hours out.
TENNIS_MAX_MIN = 36 * 60  # 36 hours — covers tomorrow's full slate
TENNIS_MIN_MIN = 5


@dataclass
class TennisModel:
    """Pregame moneyline tennis model. Same edge thesis as team sports."""

    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not file_config().models.sports.enabled:
            return None

        ticker = market.ticker or ""
        title = market.raw.get("title") or ""
        rules = market.raw.get("rules_primary") or ""

        # ----- Read structured fields -----
        # Tennis titles are sentence-shaped ("Will X win the A vs B..."),
        # so try the tennis-specific parser first; fall back to the
        # generic title / rules parsers for unusual formats.
        teams = (
            teams_from_tennis_title(title)
            or teams_from_title(title)
            or teams_from_rules(rules)
        )
        if not teams:
            log.info("tennis.skip.no_player_names",
                     ticker=ticker, title=title[:120])
            return None
        player_a, player_b = teams

        our_name = yes_side_name(market.raw)
        if not our_name:
            log.info("tennis.skip.no_yes_side", ticker=ticker)
            return None

        tip_utc = event_start_utc(market.raw)
        if tip_utc:
            mins_to_tip = (tip_utc - datetime.now(timezone.utc)).total_seconds() / 60
            if mins_to_tip < TENNIS_MIN_MIN:
                log.info("tennis.skip.too_close_to_start",
                         ticker=ticker, mins_to_tip=round(mins_to_tip, 1))
                return None
            if mins_to_tip > TENNIS_MAX_MIN:
                log.info("tennis.skip.too_far_from_start",
                         ticker=ticker, mins_to_tip=round(mins_to_tip, 1))
                return None
            date_utc_str = tip_utc.strftime("%Y-%m-%d")
        else:
            mins_to_tip = None
            date_utc_str = None

        # ----- Fair probability from book consensus -----
        fair = await fair_probability_for_tennis(
            player_a_name=player_a,
            player_b_name=player_b,
            date_utc=date_utc_str,
        )
        if not fair:
            log.info("tennis.skip.no_book_match",
                     ticker=ticker, a=player_a, b=player_b, date=date_utc_str)
            return None
        fair_a, fair_b, provider, a_full, b_full = fair

        # ----- Map our YES side to player_a / player_b probability -----
        if name_match(our_name, player_a) or name_match(our_name, a_full):
            p_yes = fair_a
        elif name_match(our_name, player_b) or name_match(our_name, b_full):
            p_yes = fair_b
        else:
            log.info("tennis.skip.our_player_unmatched",
                     ticker=ticker, our=our_name, a=a_full, b=b_full)
            return None

        p_yes = max(0.02, min(0.98, p_yes))

        # ----- WElo observe-only comparison -----
        # Independent second opinion (Weighted Elo from match history).
        # Logged for now, NOT used to size or gate trades — we want to
        # measure whether WElo agrees/disagrees with Pinnacle and which
        # is right before trusting it. welo_p is for player_a; flip if
        # our side is player_b.
        try:
            from ..welo import engine as _welo
            welo_a = _welo.win_probability(a_full, b_full)
            if welo_a is not None:
                our_is_a = (name_match(our_name, player_a)
                            or name_match(our_name, a_full))
                welo_p = welo_a if our_is_a else (1.0 - welo_a)
                log.info("welo.compare", ticker=ticker,
                         pinnacle_p=round(p_yes, 3),
                         welo_p=round(welo_p, 3),
                         delta=round(welo_p - p_yes, 3))
        except Exception:  # noqa: BLE001 - observe-only must never break trading
            pass

        # Tennis books are tighter on majors, looser on Challengers /
        # ITF. Cap confidence at 0.7 — slightly under team sports — so
        # Kelly size stays restrained when the book is thin.
        confidence = 0.70

        tip_str = f"start in {mins_to_tip:.0f}min" if mins_to_tip is not None else "no start time"
        reason = (
            f"TENNIS pregame {a_full} vs {b_full} | "
            f"book[{provider}] fair_a={fair_a:.3f} fair_b={fair_b:.3f} | "
            f"our={our_name} p_yes={p_yes:.3f} | {tip_str}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
