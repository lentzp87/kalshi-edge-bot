"""De-vigging math.

Sportsbooks bake their margin (the "vig" or "juice") into both sides of
a moneyline. To recover the implied fair probability, we:

  1. Convert each side's American odds to a raw implied probability.
  2. Normalize so the two sides sum to 1.0.

That gives us a no-vig fair probability we can compare to Kalshi's
executable price.
"""

from __future__ import annotations


def american_to_implied(odds: int | float | None) -> float | None:
    """Convert American moneyline odds to raw implied probability.

    Examples:
        -150 -> 0.6  (favored, 60% implied)
        +200 -> 0.333 (underdog, 33% implied)
    Returns None on garbage input.
    """
    if odds is None:
        return None
    try:
        o = float(odds)
    except (TypeError, ValueError):
        return None
    if o < 0:
        return abs(o) / (abs(o) + 100)
    if o > 0:
        return 100 / (o + 100)
    return None  # 0 odds is invalid


def devig_two_way(
    home_odds: int | float | None,
    away_odds: int | float | None,
) -> tuple[float, float] | None:
    """De-vig a two-way (moneyline) market.

    Returns (fair_home_prob, fair_away_prob), summing to 1.0.
    Returns None if either side can't be parsed.
    """
    h = american_to_implied(home_odds)
    a = american_to_implied(away_odds)
    if h is None or a is None:
        return None
    total = h + a
    if total <= 0:
        return None
    return h / total, a / total
