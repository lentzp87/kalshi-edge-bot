"""Edge calculation + position sizing.

Inputs:  Market + ProbabilityEstimate
Output:  TradeSignal | None

A signal only fires when:
  - the model has an opinion (estimate != None)
  - |edge| >= min_edge
  - sized position respects risk caps (sizing happens here, risk *enforces* in risk.py)
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from .config import file_config
from .kalshi_client import Market
from .models.base import ProbabilityEstimate

log = structlog.get_logger(__name__)


@dataclass
class TradeSignal:
    ticker: str
    side: str          # "yes" or "no"
    price_cents: int   # the limit price we'll post
    size_usd: float    # dollar exposure we want
    edge: float        # signed edge in probability points
    reason: str


def _kelly_size(bankroll: float, edge: float, price: float) -> float:
    """Fractional Kelly position size for a binary contract priced at `price`.

    Kelly fraction f* = edge / (1 - price)   when betting YES at price p with edge e.
    We scale by config.decision.kelly_fraction (default 0.25).
    """
    cfg = file_config().decision
    if price <= 0 or price >= 1:
        return 0.0
    full_kelly = max(edge, 0) / (1 - price)
    return bankroll * full_kelly * cfg.kelly_fraction


def evaluate(market: Market, est: ProbabilityEstimate) -> TradeSignal | None:
    cfg = file_config()
    market_p_yes = market.mid              # implied prob from mid
    edge_yes = est.p_yes - market_p_yes
    edge_no = -edge_yes

    if abs(edge_yes) < cfg.decision.min_edge:
        return None

    side = "yes" if edge_yes > 0 else "no"
    edge = edge_yes if side == "yes" else edge_no
    target_price = market.yes_ask if side == "yes" else (1 - market.yes_bid)

    # Skip low-priced entries — they're long-tail bets with bad risk/reward.
    # We want to be the favored side: only enter when the market thinks our
    # outcome is at least min_entry_price likely.
    if target_price < cfg.decision.min_entry_price:
        log.debug("decision.skip.low_price",
                  ticker=market.ticker, side=side, price=target_price,
                  min=cfg.decision.min_entry_price)
        return None

    size_usd = _kelly_size(cfg.bankroll_usd, edge * est.confidence, target_price)
    size_usd = min(size_usd, cfg.risk.max_position_size_usd)
    if size_usd < 5:  # don't bother with sub-$5 fills
        return None

    return TradeSignal(
        ticker=market.ticker,
        side=side,
        price_cents=int(round(target_price * 100)),
        size_usd=size_usd,
        edge=edge,
        reason=est.reason,
    )
