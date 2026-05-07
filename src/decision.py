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

    # Skip cheap entries — bad risk/reward and high fee drag
    if target_price < cfg.decision.min_entry_price:
        log.info("decision.skip.low_price",
                 ticker=market.ticker, side=side,
                 price=round(target_price, 3),
                 p_yes=round(p_yes, 3),
                 min=cfg.decision.min_entry_price)
        return None

    # Subtract fees + slippage. Estimate contract count at our max position
    # size to compute the right fee buffer (fees are per-contract).
    contracts_est = max(1, int(cfg.risk.max_position_size_usd / max(target_price, 0.01)))
    fee_buf = fee_buffer_pp(target_price, contracts_est)
    net_edge = gross_edge - fee_buf - _SLIPPAGE_BUFFER

    if net_edge < cfg.decision.min_edge:
        log.info("decision.skip.edge_too_small",
                 ticker=market.ticker, side=side,
                 gross=round(gross_edge, 4), fee_buf=round(fee_buf, 4),
                 net=round(net_edge, 4), min=cfg.decision.min_edge,
                 p_yes=round(p_yes, 3), price=round(target_price, 3))
        return None

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
