"""Edge calculation + position sizing.

A signal only fires when:
  - the model has an opinion (estimate != None)
  - net_edge (after fees + slippage) >= min_edge
  - executable price >= min_entry_price
  - Kelly-sized position is at least $5

Key change vs earlier version:
  Edge is now computed against the *executable* price (yes_ask for a
  YES buy, 1-yes_bid for a NO buy), NOT the midpoint. Using midpoint
  systematically overstates edge by half the spread, which was the
  root cause of "predicted edge >> realized edge" we saw before.

  We also subtract a Kalshi fee buffer and a small slippage buffer
  before applying the min-edge threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from .config import file_config
from .fee_model import fee_buffer_pp
from .kalshi_client import Market
from .models.base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Conservative slippage buffer — accounts for top-of-book moving against
# us between signal and fill. Set to 0 in paper mode (no real fills, no
# real slippage). Bump back to 0.005-0.010 before going live.
_SLIPPAGE_BUFFER = 0.0


@dataclass
class TradeSignal:
    ticker: str
    side: str          # "yes" or "no"
    price_cents: int   # the limit price we'll post
    size_usd: float    # dollar exposure we want
    edge: float        # net edge in probability points (after fees + slippage)
    reason: str


def _kelly_size(bankroll: float, edge: float, price: float) -> float:
    """Fractional Kelly position size for a binary contract priced at `price`.
    f* = edge / (1 - price), scaled by kelly_fraction.
    """
    cfg = file_config().decision
    if price <= 0 or price >= 1:
        return 0.0
    full_kelly = max(edge, 0) / (1 - price)
    return bankroll * full_kelly * cfg.kelly_fraction


def evaluate(market: Market, est: ProbabilityEstimate) -> TradeSignal | None:
    cfg = file_config()

    # Executable prices — what we ACTUALLY pay to buy each side.
    yes_ask = market.effective_yes_ask
    yes_bid = market.effective_yes_bid
    if yes_ask <= 0 or yes_bid <= 0 or yes_ask >= 1 or yes_bid >= 1:
        log.info("decision.skip.bad_book", ticker=market.ticker,
                 yes_ask=yes_ask, yes_bid=yes_bid)
        return None
    no_ask = 1.0 - yes_bid  # buying NO at p means selling YES at (1-p)

    # Gross edges (vs executable prices, not midpoint)
    p_yes = est.p_yes
    gross_yes = p_yes - yes_ask
    gross_no = (1 - p_yes) - no_ask  # equivalent to: yes_bid - p_yes

    # Pick the side with positive edge
    if gross_yes >= gross_no:
        side = "yes"
        target_price = yes_ask
        gross_edge = gross_yes
    else:
        side = "no"
        target_price = no_ask
        gross_edge = gross_no

    # Skip cheap entries — bad risk/reward and high fee drag.
    # Side-specific floor: pick the YES or NO knob if set, else fall
    # back to the legacy single `min_entry_price`. Keeps room for an
    # evidence-based asymmetric policy once we have enough samples.
    side_floor = (
        cfg.decision.min_entry_price_yes if side == "yes"
        else cfg.decision.min_entry_price_no
    )
    min_floor = side_floor if side_floor is not None else cfg.decision.min_entry_price
    if target_price < min_floor:
        log.info("decision.skip.low_price",
                 ticker=market.ticker, side=side,
                 price=round(target_price, 3),
                 p_yes=round(p_yes, 3),
                 min=min_floor)
        return None

    # Skip low-probability sides. p_yes is the model's prob that the
    # YES side wins; if we picked NO, our side's true prob is 1-p_yes.
    # Filter on the side WE'RE actually betting on.
    our_side_p = p_yes if side == "yes" else (1 - p_yes)
    if our_side_p < cfg.decision.min_p_yes:
        log.info("decision.skip.p_yes_too_low",
                 ticker=market.ticker, side=side,
                 our_side_p=round(our_side_p, 3),
                 min=cfg.decision.min_p_yes)
        return None
    # Skip overconfident model picks. 2026-05-19 data: 80%+ confidence
    # bucket had 33% wr / -$285 over N=36 trades. Capping where the
    # model has historically been most wrong.
    max_p = cfg.decision.max_p_yes
    if max_p is not None and our_side_p > max_p:
        log.info("decision.skip.p_yes_too_high",
                 ticker=market.ticker, side=side,
                 our_side_p=round(our_side_p, 3),
                 max=max_p)
        return None

    # ----- Per-market fee-adjusted edge gate -----
    # ChatGPT pushback: a global min_edge ignores that the fee curve
    # varies by price (peaks near 0.50). Compute a per-market required
    # edge that includes both legs of fees, half the spread we have to
    # cross, slippage cushion, and a safety margin. Then check raw
    # gross_edge against THIS bar, not against a flat threshold.
    contracts_est = max(1, int(cfg.risk.max_position_size_usd / max(target_price, 0.01)))
    entry_fee_pp = fee_buffer_pp(target_price, contracts_est)
    # Estimate exit at the same price (worst case). Real exit may be
    # cheaper/dearer depending on direction, but this is conservative.
    exit_fee_pp = entry_fee_pp
    # Half-spread we have to cross to enter. Already captured in
    # executable price vs mid, but explicit factor keeps the math honest
    # if the executable_price calculation ever changes.
    spread = market.effective_yes_ask - market.effective_yes_bid
    half_spread_pp = (spread / 2) / max(target_price, 0.01)
    # Configurable safety margin so we don't fire on knife-edge edges.
    safety_pp = cfg.decision.required_edge_safety_pp
    required_edge = (
        entry_fee_pp + exit_fee_pp + half_spread_pp
        + _SLIPPAGE_BUFFER + safety_pp
    )
    net_edge = gross_edge - entry_fee_pp - exit_fee_pp - _SLIPPAGE_BUFFER
    # Two-tier gate:
    #   1. Raw gross_edge must exceed required_edge — covers all costs.
    #   2. Net edge must clear cfg.decision.min_edge — paper-mode floor.
    if gross_edge < required_edge:
        log.info("decision.skip.edge_below_required",
                 ticker=market.ticker, side=side,
                 gross=round(gross_edge, 4),
                 required=round(required_edge, 4),
                 entry_fee=round(entry_fee_pp, 4),
                 exit_fee=round(exit_fee_pp, 4),
                 half_spread=round(half_spread_pp, 4),
                 safety=round(safety_pp, 4),
                 p_yes=round(p_yes, 3), price=round(target_price, 3))
        return None
    if net_edge < cfg.decision.min_edge:
        log.info("decision.skip.edge_too_small",
                 ticker=market.ticker, side=side,
                 gross=round(gross_edge, 4), fee_buf=round(entry_fee_pp, 4),
                 net=round(net_edge, 4), min=cfg.decision.min_edge,
                 p_yes=round(p_yes, 3), price=round(target_price, 3))
        return None
    # Keep original variable name for fee_buf in the rest of the file
    fee_buf = entry_fee_pp

    # Kelly sizing on net edge, scaled by model confidence
    size_usd = _kelly_size(cfg.bankroll_usd, net_edge * est.confidence, target_price)
    size_usd = min(size_usd, cfg.risk.max_position_size_usd)
    if size_usd < 5:
        log.info("decision.skip.size_too_small",
                 ticker=market.ticker, size_usd=round(size_usd, 2),
                 net_edge=round(net_edge, 4))
        return None

    log.info("decision.signal_fired",
             ticker=market.ticker, side=side,
             price=round(target_price, 3),
             size_usd=round(size_usd, 2),
             net_edge=round(net_edge, 4),
             p_yes=round(p_yes, 3))
    return TradeSignal(
        ticker=market.ticker,
        side=side,
        price_cents=int(round(target_price * 100)),
        size_usd=size_usd,
        edge=net_edge,
        reason=(
            f"{est.reason} | net_edge={net_edge:+.3f} "
            f"(gross {gross_edge:+.3f} - fees {fee_buf:.3f} - slip {_SLIPPAGE_BUFFER:.3f}) "
            f"side={side} entry={target_price:.2f}"
        ),
    )
