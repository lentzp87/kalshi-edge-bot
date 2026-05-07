"""Tennis match moneyline model.

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

Pipeline
--------
  1. Parse Kalshi ticker -> (player_a, player_b, our_player, date_utc).
     Series: KXWTAMATCH (WTA) or KXATPMATCH (ATP).
     Event:  YYMMM<DD><player_a_token><player_b_token>
             where each token is 2-4 letters (typically last-name prefix).
  2. Search across all active tennis sport keys for an event whose
     home_team / away_team match player_a_token / player_b_token.
  3. De-vig moneyline median across books.
  4. Return p_yes for our_player.

The decision layer (src/decision.py) handles fee math + executable
price + Kelly sizing — same as for team sports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from ..config import file_config
from ..kalshi_client import Market
from ..odds_provider import fair_probability_for_tennis
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Series prefixes we handle here.
TENNIS_SERIES: set[str] = {"KXWTAMATCH", "KXATPMATCH"}


_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# Market ticker:  KX(WTA|ATP)MATCH-<event_code>-<our_player_token>
_TICKER_RE = re.compile(
    r"""^
        (?P<series>KX(?:WTA|ATP)MATCH) -
        (?P<eventcode>[A-Z0-9]+) -
        (?P<our>[A-Z0-9]+)
        $""", re.VERBOSE,
)


@dataclass
class _ParsedTennisTicker:
    series: str
    player_a_token: str
    player_b_token: str
    our_player_token: str
    match_date_utc: str | None = None


def _split_event_code(code: str) -> tuple[str, str, str | None] | None:
    """Pull (player_a, player_b, YYYY-MM-DD) out of the event-code segment.

    Examples observed on Kalshi:
        '26MAY04TOWSRA'   -> (TOW, SRA, 2026-05-04)
        '26MAY041430ALCM' -> (ALC, M??, 2026-05-04) — with a time block

    The pattern is YY + MMM + (DD or DDHHMM) + <letters>. We greedy-eat
    the leading digits and split the trailing letter run roughly in half.
    """
    m = re.match(
        r"^(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<rest>\d+)(?P<players>[A-Z]+)$",
        code,
    )
    if not m:
        return None
    players = m.group("players")
    n = len(players)
    if n < 4:
        return None
    # Try splitting at common token lengths. Tennis player tokens are
    # almost always 3 letters, occasionally 2 or 4.
    pair = None
    for left in (3, 4, 2):
        right = n - left
        if 2 <= right <= 4:
            pair = (players[:left], players[left:])
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


def parse_ticker(market: Market) -> _ParsedTennisTicker | None:
    market_ticker = market.ticker or ""
    m = _TICKER_RE.match(market_ticker)
    if not m:
        return None
    series = m.group("series")
    if series not in TENNIS_SERIES:
        return None
    our = m.group("our")

    event_ticker = market.raw.get("event_ticker", "") or ""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None
    parsed = _split_event_code(parts[1])
    if not parsed:
        return None
    a, b, date_utc = parsed
    return _ParsedTennisTicker(
        series=series,
        player_a_token=a,
        player_b_token=b,
        our_player_token=our,
        match_date_utc=date_utc,
    )


@dataclass
class TennisModel:
    """Pregame moneyline tennis model. Same edge thesis as team sports."""

    enabled: bool = True

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        # Honor the same kill switch as the rest of the sports family
        if not file_config().models.sports.enabled:
            return None

        parsed = parse_ticker(market)
        if not parsed:
            log.info("tennis.skip.parse_failed",
                     ticker=market.ticker,
                     event_ticker=market.raw.get("event_ticker", ""))
            return None

        fair = await fair_probability_for_tennis(
            player_a_token=parsed.player_a_token,
            player_b_token=parsed.player_b_token,
            date_utc=parsed.match_date_utc,
        )
        if not fair:
            log.info("tennis.skip.no_book_match",
                     ticker=market.ticker,
                     a=parsed.player_a_token, b=parsed.player_b_token,
                     date=parsed.match_date_utc)
            return None
        fair_a, fair_b, provider, a_full, b_full = fair

        # Map to our side: which of (a, b) is the market's "yes" side?
        our = parsed.our_player_token.upper()
        if our == parsed.player_a_token.upper():
            p_yes = fair_a
        elif our == parsed.player_b_token.upper():
            p_yes = fair_b
        else:
            # Last resort: substring match against the resolved full names
            from ..odds_provider import _player_match
            if _player_match(our, a_full):
                p_yes = fair_a
            elif _player_match(our, b_full):
                p_yes = fair_b
            else:
                log.info("tennis.skip.our_player_unmatched",
                         ticker=market.ticker, our=our,
                         a_full=a_full, b_full=b_full)
                return None

        p_yes = max(0.02, min(0.98, p_yes))

        # Tennis books are tighter on majors, looser on Challengers /
        # ITF. Cap confidence at 0.7 — slightly under team sports — so
        # Kelly size stays restrained when the book is thin.
        confidence = 0.70

        reason = (
            f"TENNIS pregame {a_full} vs {b_full} | "
            f"book[{provider}] fair_a={fair_a:.3f} fair_b={fair_b:.3f} | "
            f"our={parsed.our_player_token} p_yes={p_yes:.3f}"
        )
        return ProbabilityEstimate(p_yes=p_yes, confidence=confidence, reason=reason)
