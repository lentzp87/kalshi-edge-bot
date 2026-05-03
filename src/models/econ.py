"""Econ-data model.

Strategy: for CPI / NFP / unemployment / FOMC markets, compare:
  - market implied prob
  - consensus forecast distribution from BLS/BEA/Fed releases

Stub for now; fills in once you've decided on your data feed (e.g., FRED API).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import file_config
from ..kalshi_client import Market
from .base import ProbabilityEstimate


@dataclass
class EconModel:
    enabled: bool = False

    def __post_init__(self) -> None:
        self.enabled = file_config().models.econ.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None
        # TODO: pull consensus + dispersion from your data feed and compute p_yes.
        return None
