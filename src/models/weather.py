"""Weather model.

Strategy: Kalshi has many "high temp at <city> on <date> >= X°F" markets.
For each, pull the latest forecast (NWS for US, Open-Meteo as fallback),
extract a forecast distribution, and integrate the probability that the
realized value exceeds the strike.

This stub returns None (no opinion) so the bot is safe to run before
you wire real forecast data. Replace `_forecast_distribution` with a
real call to your forecast provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from ..config import file_config
from ..kalshi_client import Market
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


@dataclass
class WeatherModel:
    enabled: bool = True

    def __post_init__(self) -> None:
        self.enabled = file_config().models.weather.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None

        parsed = self._parse_title(market.title)
        if not parsed:
            return None

        forecast_mean, forecast_sd = await self._forecast_distribution(
            location=parsed["location"], variable=parsed["variable"]
        )
        if forecast_mean is None:
            return None

        # Probability that forecast > strike, assuming normal forecast errors.
        # Replace with empirical CDF when you have history.
        from math import erf, sqrt

        z = (forecast_mean - parsed["strike"]) / max(forecast_sd, 1e-6)
        p_yes = 0.5 * (1 + erf(z / sqrt(2)))
        return ProbabilityEstimate(
            p_yes=p_yes,
            confidence=0.5,  # raise once you have backtest evidence
            reason=(
                f"{parsed['variable']} at {parsed['location']} "
                f"forecast={forecast_mean:.1f}±{forecast_sd:.1f} "
                f"strike={parsed['strike']} -> p_yes={p_yes:.3f}"
            ),
        )

    # ---- stubs you'll replace ----

    @staticmethod
    def _parse_title(title: str) -> dict | None:
        """Very light parser. Real version should handle Kalshi's title formats."""
        m = re.search(r"(?P<var>high|low|temp|rain).*?(?P<num>-?\d+(?:\.\d+)?)", title.lower())
        if not m:
            return None
        return {
            "variable": m.group("var"),
            "strike": float(m.group("num")),
            "location": "unknown",  # extract from market.raw['event_ticker'] in real impl
        }

    async def _forecast_distribution(
        self, *, location: str, variable: str
    ) -> tuple[float | None, float]:
        """Return (mean, std-dev) for the forecast variable.

        Until you wire a real provider, return (None, 0) so the model
        emits no signal and the bot stays out of these markets.
        """
        return None, 0.0
