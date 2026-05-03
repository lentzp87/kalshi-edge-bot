"""Crypto model — placeholder.

Crypto price-level markets (e.g. BTC > X by Friday) are nearly martingales,
so the implied probability is usually correct. Look instead for:
  - Mispriced volatility (Kalshi's straddles vs. options market vol)
  - Time-decay near expiry where order flow temporarily distorts prices
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import file_config
from ..kalshi_client import Market
from .base import ProbabilityEstimate


@dataclass
class CryptoModel:
    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = file_config().models.crypto.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None
        return None
