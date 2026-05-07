"""Light-touch injury / lineup check.

KEY INSIGHT: the sportsbook consensus already prices in known injuries.
If Embiid is out and the book has shifted PHI from -3 to +2, that move
already reflects the absence. Our injury filter should NOT re-deduct
anything for normal, market-known absences — we'd be double-counting.

So the filter only fires when we expect the *book* might be stale or
the chaos is unusual:

  * NBA / NFL: skip only if >=3 starters are listed "out" on one team
    (signals lineup is in flux beyond a normal star scratching).
  * MLB / NHL: trust the book entirely. Day-to-day relievers and bench
    guys litter every event; the book has them priced.

This intentionally allows trading even when a single star is out — the
edge thesis is that Kalshi's price diverges from the de-vigged book
consensus, and the book already accounts for the star.
"""

from __future__ import annotations

from .espn_client import fetch_summary

# How many players listed as "out" on a single team before we treat it
# as lineup chaos (not just a normal star scratch the book has priced).
_OUT_THRESHOLD_BY_SPORT: dict[str, int] = {
    "nba": 3,   # one star out is fine; 3 starters out = real chaos
    "nfl": 4,   # bigger rosters, more outs are normal
    "mlb": 99,  # disabled — book handles MLB injuries; rosters are huge
    "nhl": 99,  # disabled — book handles NHL injuries; rosters are huge
}

# Only "out" counts. Questionable / day-to-day / doubtful are pre-game
# noise that the book consensus already incorporates.
_BLOCKING_STATUSES: set[str] = {"out"}


async def has_risky_injury(sport_key: str, espn_event_id: str) -> tuple[bool, list[str]]:
    """Return (any_risky, list of "TEAM Player (Status)" strings).

    Only blocks when a single team has the sport-specific threshold of
    "out" players — a heuristic for lineup chaos that the sportsbook
    line may not yet fully reflect. Returns (False, []) otherwise.
    """
    threshold = _OUT_THRESHOLD_BY_SPORT.get(sport_key, 99)
    if threshold >= 99:
        # Trust sportsbook; don't fetch the summary at all (saves a call)
        return False, []

    s = await fetch_summary(sport_key, espn_event_id)
    if not s:
        return False, []
    teams = s.get("injuries") or []
    per_team_out: dict[str, list[str]] = {}
    for team_block in teams:
        team_abbr = (team_block.get("team") or {}).get("abbreviation", "?")
        for inj in team_block.get("injuries") or []:
            status = (inj.get("status") or "").lower()
            if status not in _BLOCKING_STATUSES:
                continue
            player = (inj.get("athlete") or {}).get("displayName", "?")
            per_team_out.setdefault(team_abbr, []).append(f"{team_abbr} {player} ({status})")
    risky: list[str] = []
    for team, players in per_team_out.items():
        if len(players) >= threshold:
            risky.extend(players)
    return (len(risky) > 0), risky
