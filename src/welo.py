"""Weighted Elo (WElo) for tennis — an INDEPENDENT probability model.

Why this exists
---------------
Today the bot's tennis "model" is just Pinnacle's line de-vigged. If
Pinnacle is wrong, we have no way to know. WElo gives us a *second,
independent* opinion built from match results — so we can bet when our
own number AND Kalshi disagree, rather than blindly echoing one book.

Academic basis: Angelini, Candila & De Angelis (2022, EJOR) — Weighted
Elo, validated out-of-sample on 60k+ ATP/WTA matches, documented
~3.0-3.6% ROI vs sharp bookmaker lines.

How WElo differs from plain Elo
-------------------------------
Plain Elo updates a rating by `K * (1 - expected)` on a win. WElo
weights the update by the *scoreline*: instead of a binary 1 for a win,
the winner's "score" is their share of games won. A 6-0 6-0 demolition
(games 12-0, share 1.00) moves ratings far more than a 7-6 7-6 squeaker
(games ~14-12, share ~0.54). This captures conviction the bookmaker's
binary win/loss partially misses.

  update = K * (games_share - expected_win_prob)

Surface-specific tracks (clay / grass / hard) are maintained alongside
an `overall` track. `win_probability(a, b, surface)` uses the surface
track when given, else `overall`.

Seeding
-------
Ratings are bootstrapped by replaying recent ATP+WTA match history from
Jeff Sackmann's public dataset (github.com/JeffSackmann). `seed()` is
launched as a background task at startup; until it completes,
`win_probability` returns None and callers fall back to the existing
Pinnacle path. READ-ONLY / observe mode — WElo does not yet influence
any trade; it is logged alongside the Pinnacle number so we can measure
whether it adds signal before trusting it.
"""

from __future__ import annotations

import re
import time

import httpx
import structlog

log = structlog.get_logger(__name__)

BASE_RATING = 1500.0
K_FACTOR = 32.0
_SACKMANN = {
    "atp": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{y}.csv",
    "wta": "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{y}.csv",
}
# How many recent years of history to replay. 3 years is plenty for
# ratings to converge while staying current on form.
SEED_YEARS = (2024, 2025, 2026)

_TIEBREAK_RE = re.compile(r"\([^)]*\)")


def _norm_name(name: str) -> str:
    """Canonical key for a player. Handles 'First Last', 'Last, First',
    punctuation, case — so the Sackmann dataset and Pinnacle/Kalshi
    names resolve to the same key. Returns sorted lowercase tokens.
    """
    if not name:
        return ""
    n = name.replace(",", " ").lower()
    n = re.sub(r"[^a-z\s]", " ", n)
    tokens = sorted(t for t in n.split() if len(t) > 1)
    return " ".join(tokens)


def _parse_score(score: str) -> tuple[int, int] | None:
    """Parse a tennis score string into (winner_games, loser_games).
    Returns None for retirements / walkovers / unparseable scores.
    """
    if not score:
        return None
    s = _TIEBREAK_RE.sub("", score)  # drop tiebreak detail "(7)"
    wg = lg = 0
    for chunk in s.split():
        if "-" not in chunk:
            continue  # "RET", "W/O", "Def.", etc.
        a, _, b = chunk.partition("-")
        try:
            wg += int(a)
            lg += int(b)
        except ValueError:
            continue
    if wg + lg == 0:
        return None
    return wg, lg


def _surface_key(surface: str | None) -> str:
    s = (surface or "").strip().lower()
    if s in ("clay", "grass", "hard"):
        return s
    return "overall"


class WeloEngine:
    """Per-(track, player) Elo ratings. `track` is 'overall' or a
    surface ('clay'/'grass'/'hard')."""

    def __init__(self) -> None:
        self._ratings: dict[str, dict[str, float]] = {}
        self.seeded: bool = False
        self.n_matches: int = 0
        self.last_seed_ts: float = 0.0

    def _get(self, track: str, player: str) -> float:
        return self._ratings.setdefault(track, {}).get(player, BASE_RATING)

    def _set(self, track: str, player: str, r: float) -> None:
        self._ratings.setdefault(track, {})[player] = r

    def update(self, winner: str, loser: str, surface: str | None,
               games_won: int, games_lost: int) -> None:
        """Apply one match result. Updates both the `overall` track and
        the surface track. Winner's outcome is their games share, not a
        binary 1 — that is the 'weighted' in Weighted Elo."""
        w = _norm_name(winner)
        l = _norm_name(loser)
        if not w or not l or w == l:
            return
        total = games_won + games_lost
        share = games_won / total if total else 1.0
        # Clamp so a freak bagel doesn't over-swing; keep it informative.
        share = max(0.5, min(1.0, share))
        for track in ("overall", _surface_key(surface)):
            ra = self._get(track, w)
            rb = self._get(track, l)
            exp_w = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
            delta = K_FACTOR * (share - exp_w)
            self._set(track, w, ra + delta)
            self._set(track, l, rb - delta)
        self.n_matches += 1

    def win_probability(self, player_a: str, player_b: str,
                        surface: str | None = None) -> float | None:
        """P(player_a beats player_b). None if the engine isn't seeded
        or we have no rating history for one of the players."""
        if not self.seeded:
            return None
        a = _norm_name(player_a)
        b = _norm_name(player_b)
        if not a or not b:
            return None
        track = _surface_key(surface)
        # Require at least an overall rating for both players. If a
        # surface track is requested but a player has no surface
        # history, fall back to overall for that comparison.
        overall = self._ratings.get("overall", {})
        if a not in overall or b not in overall:
            return None
        if track != "overall" and (a in self._ratings.get(track, {})
                                   and b in self._ratings.get(track, {})):
            ra, rb = self._get(track, a), self._get(track, b)
        else:
            ra, rb = self._get("overall", a), self._get("overall", b)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def player_count(self) -> int:
        return len(self._ratings.get("overall", {}))


# Module-level singleton.
engine = WeloEngine()


async def seed(years: tuple[int, ...] = SEED_YEARS) -> None:
    """Download recent ATP+WTA match history and replay it to build
    ratings. Safe to call once at startup. Logs loudly; on total
    failure the engine simply stays unseeded and win_probability
    returns None (callers fall back to Pinnacle)."""
    rows: list[tuple[str, str, str, str]] = []  # (winner, loser, surface, score)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for tour, url_tpl in _SACKMANN.items():
                for y in years:
                    url = url_tpl.format(y=y)
                    try:
                        r = await client.get(url)
                        if r.status_code != 200:
                            continue
                        rows.extend(_parse_csv(r.text))
                    except Exception as e:  # noqa: BLE001
                        log.warning("welo.seed.fetch_failed",
                                    tour=tour, year=y, err=str(e)[:120])
    except Exception:
        log.exception("welo.seed.error")
        return

    if not rows:
        log.warning("welo.seed.no_data",
                    note="no match history fetched; WElo stays unseeded")
        return

    # Sackmann files are roughly chronological within a year and we
    # fetch years in ascending order — replay in that order.
    for winner, loser, surface, score in rows:
        parsed = _parse_score(score)
        if parsed is None:
            continue
        engine.update(winner, loser, surface, parsed[0], parsed[1])

    engine.seeded = True
    engine.last_seed_ts = time.time()
    log.info("welo.seeded", matches=engine.n_matches,
             players=engine.player_count())


def _parse_csv(text: str) -> list[tuple[str, str, str, str]]:
    """Pull (winner_name, loser_name, surface, score) from a Sackmann
    match CSV. Uses the header row to find columns (column order has
    been stable for years but we don't hard-code indices)."""
    import csv
    import io
    out: list[tuple[str, str, str, str]] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        w = row.get("winner_name")
        l = row.get("loser_name")
        if not w or not l:
            continue
        out.append((w, l, row.get("surface") or "", row.get("score") or ""))
    return out
