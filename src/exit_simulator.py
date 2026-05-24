"""Counterfactual exit-policy simulator for closed trades.

Given a closed trade with known entry fill price, contracts, observed
max/min mid during the hold, and a settlement P&L, simulate what the
realized P&L would have been under alternative exit policies:

  - actual: as-recorded pnl_usd
  - tp_only_20: take-profit at fill * 1.20 if max_mid touched it
  - sl_only_12: stop-loss at fill * 0.88 if min_mid touched it
  - time_only: hold to settlement (use settlement_pnl_usd)
  - tp_and_sl_20_12: both thresholds active; if both hit, approximate
    "which fired first" by picking the exit closer to fill (smaller |P&L|)

For simulated exits, P&L = (exit - fill) * contracts - 2 * fee_per_contract_dollars(mid) * contracts
where mid = (fill + exit) / 2 (used as a fee-estimate price).

If the inputs needed for a policy are missing (e.g. settlement_pnl_usd
is NULL for an in-progress game, or max_mid_during_hold was never
written), the trade is skipped for that policy (returns None).
"""

from __future__ import annotations

from typing import Optional

from .fee_model import fee_per_contract_dollars


TP_MULT = 1.20  # take-profit at +20%
SL_MULT = 0.88  # stop-loss at -12%

POLICIES = (
    "actual",
    "tp_only_20",
    "sl_only_12",
    "time_only",
    "tp_and_sl_20_12",
    "exit_75min",
)

# Hard-exit cap (minutes). The "exit_75min" policy keeps take-profit as
# the bot does today, but additionally closes any position still open
# at this age. Research (hold-duration analysis) suggests a trade open
# this long has likely missed its inflection point — but we backtest it
# from instrumented data rather than flipping the live setting blind.
HARD_EXIT_MINUTES = 75


def _sim_pnl(fill: float, exit_price: float, contracts: float) -> float:
    """P&L for a simulated exit, net of entry+exit fees estimated at midpoint."""
    mid = (fill + exit_price) / 2.0
    fee_each = fee_per_contract_dollars(mid) * contracts
    return (exit_price - fill) * contracts - 2.0 * fee_each


def simulate_policy(trade: dict, policy: str) -> Optional[float]:
    """Return the simulated P&L for `trade` under `policy`, or None to skip.

    `trade` must have `fill_price`, `contracts`, plus the optional
    `max_mid_during_hold`, `min_mid_during_hold`, `settlement_pnl_usd`,
    and (for the actual policy) `pnl_usd`.
    """
    if policy == "actual":
        pnl = trade.get("pnl_usd")
        return float(pnl) if pnl is not None else None

    fill = trade.get("fill_price")
    contracts = trade.get("contracts")
    if fill is None or contracts is None or contracts <= 0:
        return None
    fill = float(fill)
    contracts = float(contracts)

    settlement = trade.get("settlement_pnl_usd")
    settlement_val = float(settlement) if settlement is not None else None
    max_mid = trade.get("max_mid_during_hold")
    min_mid = trade.get("min_mid_during_hold")

    tp_price = fill * TP_MULT
    sl_price = fill * SL_MULT
    tp_hit = max_mid is not None and float(max_mid) >= tp_price
    sl_hit = min_mid is not None and float(min_mid) <= sl_price

    if policy == "tp_only_20":
        if tp_hit:
            return _sim_pnl(fill, tp_price, contracts)
        if settlement_val is not None:
            return settlement_val
        return None

    if policy == "sl_only_12":
        if sl_hit:
            return _sim_pnl(fill, sl_price, contracts)
        if settlement_val is not None:
            return settlement_val
        return None

    if policy == "time_only":
        return settlement_val

    if policy == "tp_and_sl_20_12":
        if tp_hit and sl_hit:
            tp_pnl = _sim_pnl(fill, tp_price, contracts)
            sl_pnl = _sim_pnl(fill, sl_price, contracts)
            # Cannot disambiguate temporal order from min/max alone;
            # whichever exit lies closer to fill (smaller |P&L|) is the
            # better proxy for "fired first".
            return tp_pnl if abs(tp_pnl) <= abs(sl_pnl) else sl_pnl
        if tp_hit:
            return _sim_pnl(fill, tp_price, contracts)
        if sl_hit:
            return _sim_pnl(fill, sl_price, contracts)
        return settlement_val

    if policy == "exit_75min":
        # "Keep take-profit, but hard-cap any trade still open at 75 min."
        #  - Trade that closed BEFORE 75 min (hit TP, or its own time
        #    exit): the 75-min rule never fires — keep actual P&L.
        #  - Trade open >= 75 min: the rule fires — P&L is the simulated
        #    exit at the snapshotted mid_at_75min.
        #  - mid_at_75min NULL (old trade, or never instrumented): can't
        #    simulate — skip.
        held = _hold_minutes(trade.get("opened_ts"), trade.get("closed_ts"))
        if held is not None and held < HARD_EXIT_MINUTES:
            pnl = trade.get("pnl_usd")
            return float(pnl) if pnl is not None else None
        mid75 = trade.get("mid_at_75min")
        if mid75 is None:
            return None
        return _sim_pnl(fill, float(mid75), contracts)

    return None


def _hold_minutes(opened_ts, closed_ts) -> Optional[float]:
    """Hold duration in minutes from two ISO timestamps, or None."""
    if not opened_ts or not closed_ts:
        return None
    from datetime import datetime
    try:
        o = datetime.fromisoformat(str(opened_ts).replace("Z", "+00:00"))
        c = datetime.fromisoformat(str(closed_ts).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return (c - o).total_seconds() / 60.0


def aggregate_exit_policies(trades: list[dict]) -> list[dict]:
    """Aggregate per-policy stats across `trades`. Returns rows shaped like
    `_bucket_aggregate` output: {key, n, wins, pnl, avg_pnl, win_rate}.
    """
    out: list[dict] = []
    for policy in POLICIES:
        n = 0
        wins = 0
        total = 0.0
        for t in trades:
            pnl = simulate_policy(t, policy)
            if pnl is None:
                continue
            n += 1
            total += pnl
            if pnl > 0:
                wins += 1
        out.append({
            "key": policy,
            "n": n,
            "wins": wins,
            "pnl": round(total, 2),
            "avg_pnl": round(total / n, 2) if n else 0.0,
            "win_rate": round(wins / n, 3) if n else 0.0,
        })
    return out


def edge_bucket(gross_edge: Optional[float]) -> str:
    """Bucket a gross_edge value (in pp; e.g. 0.025 = 2.5pp) by absolute magnitude.

    Returns "?" if `gross_edge` is None.
    """
    if gross_edge is None:
        return "?"
    pp = abs(float(gross_edge)) * 100.0
    if pp < 1.0: return "0-1pp"
    if pp < 2.0: return "1-2pp"
    if pp < 3.0: return "2-3pp"
    if pp < 5.0: return "3-5pp"
    return "5pp+"
