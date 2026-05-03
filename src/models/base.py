"""Common interface every probability model must implement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..kalshi_client import Market


@dataclass
class ProbabilityEstimate:
    """A model's verdict on the YES side of a market."""

    p_yes: float                 # 0.0 - 1.0
    confidence: float            # 0.0 - 1.0; lowers Kelly sizing when low
    reason: str                  # human-readable explanation; logged for review


class Model(Protocol):
    """All models implement this. Return None to mean 'no opinion'."""

    enabled: bool

    async def estimate(self, market: Market) -> ProbabilityEstimate | None: ...
