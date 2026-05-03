"""Sports model — disabled by default.

Sports lines in liquid markets are extremely efficient. The realistic edge
opportunity here is *cross-book arbitrage* (Kalshi vs. a sportsbook with a
softer line), not raw probability modeling.

Leave this disabled until you have:
  - A pricing source you trust more than Kalshi (e.g. sharp-book consensus)
  - A way to detect Kalshi lagging that source by enough to overcome fees
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import file_config
from ..kalshi_client import Market
from .base import ProbabilityEstimate


@dataclass
class SportsModel:
    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = file_config().models.sports.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None
        # TODO: pull consensus implied probability from a reference source,
        # then return only when |kalshi - reference| > threshold + fees.
        return None
