"""Light-touch injury / lineup check.

Uses ESPN's per-event `injuries` field. We only flag a game as risky
when a player listed as STAR-level (we approximate by `status in
{Out, Doubtful, Questionable}` for any roster spot since ESPN doesn't
expose star-rating). Conservative: skip the game if any player is
listed in those statuses, so we avoid trading into late lineup chaos.

This is a v1 heuristic. A more sophisticated version would weight by
the player's projected impact on win probability (RAPM, EPA, etc.).
"""

from __future__ import annotations

from .espn_client import fetch_summary

# Statuses we treat as "uncertain enough to skip"
RISKY_STATUSES: set[str] = {
    "out", "doubtful", "questionable", "day-to-day", "game-time decision",
}


async def has_risky_injury(sport_key: str, espn_event_id: str) -> tuple[bool, list[str]]:
    """Return (any_risky, list of "TEAM Player (Status)" strings).

    Returns (False, []) if ESPN gives us no injuries data — that's not
    proof of clean roster, but we don't want to skip every game when ESPN
    just hasn't populated injuries yet. v1 trade-off favors more activity.
    """
    s = await fetch_summary(sport_key, espn_event_id)
    if not s:
        return False, []
    teams = s.get("injuries") or []
    risky: list[str] = []
    for team_block in teams:
        team_abbr = (team_block.get("team") or {}).get("abbreviation", "?")
        for inj in team_block.get("injuries") or []:
            status = (inj.get("status") or "").lower()
            if status in RISKY_STATUSES:
                player = (inj.get("athlete") or {}).get("displayName", "?")
                risky.append(f"{team_abbr} {player} ({status})")
    return (len(risky) > 0), risky
