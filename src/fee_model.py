"""Kalshi fee model.

Kalshi's published taker fee schedule rounds approximately to:

    fee_per_contract_cents = round_up(7 * price_dollars * (1 - price_dollars))

Where `price_dollars` is the contract price in dollars (0.01 - 0.99).
That formula peaks near $0.0175/contract at 50c and goes to ~0 at the
extremes. For a 100-contract trade, fees range $0.07 - $1.75.

We use this to subtract a fee buffer from gross edge, so we never
fire a trade whose edge gets eaten by fees.
"""

from __future__ import annotations

import math


def fee_per_contract_dollars(price: float) -> float:
    """Approximate Kalshi taker fee per contract, in dollars.

    The exact schedule is published by Kalshi and rounds to the cent;
    we use a close-form approximation that matches the published table
    within +/- $0.01 across the 0.01-0.99 range.
    """
    p = max(0.01, min(0.99, float(price)))
    raw = 0.07 * p * (1.0 - p)
    # Round up to the nearest cent (Kalshi rounds in their favor).
    return math.ceil(raw * 100) / 100


def fee_buffer_pp(price: float, contracts: int) -> float:
    """Total fee as a fraction of *notional* (size_usd) — i.e. how many
    points of edge the fee eats.

    This is what the decision engine should subtract from gross edge.
    Example: at 0.60 with 67 contracts ($40 size), fee buffer ~= 0.028
    (2.8 percentage points).
    """
    if contracts <= 0 or price <= 0 or price >= 1:
        return 0.0
    fee_total = fee_per_contract_dollars(price) * contracts
    notional = price * contracts
    return fee_total / notional if notional > 0 else 0.0
