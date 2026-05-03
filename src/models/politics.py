"""Politics model — placeholder.

For headline events (election results, confirmation votes, debate winners)
your edge will come from short-window order-flow imbalances, not from
out-modeling 538/Polymarket. Disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import file_config
from ..kalshi_client import Market
from .base import ProbabilityEstimate


@dataclass
class PoliticsModel:
    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = file_config().models.politics.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None
        return None
