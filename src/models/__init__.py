"""Probability models, registered by Kalshi market category."""

from .base import Model, ProbabilityEstimate
from .crypto import CryptoModel
from .econ import EconModel
from .politics import PoliticsModel
from .sports import SportsModel
from .weather import WeatherModel

# Kalshi categories map to model classes.
# Add new mappings as you build new models.
REGISTRY: dict[str, type[Model]] = {
    "Weather": WeatherModel,
    "Climate and Weather": WeatherModel,
    "Economics": EconModel,
    "Sports": SportsModel,
    "Crypto": CryptoModel,
    "Politics": PoliticsModel,
}


def model_for_category(category: str) -> Model | None:
    cls = REGISTRY.get(category)
    return cls() if cls else None


__all__ = ["Model", "ProbabilityEstimate", "model_for_category", "REGISTRY"]
