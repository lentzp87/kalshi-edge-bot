"""Probability models, registered by Kalshi market category."""

from .base import Model, ProbabilityEstimate
from .crypto import CryptoModel
from .econ import EconModel
from .politics import PoliticsModel
from .sports import SportsModel

# Kalshi categories map to model classes.
REGISTRY: dict[str, type[Model]] = {
    "Sports": SportsModel,
    "Economics": EconModel,
    "Crypto": CryptoModel,
    "Politics": PoliticsModel,
}


def model_for_category(category: str) -> Model | None:
    cls = REGISTRY.get(category)
    return cls() if cls else None


__all__ = ["Model", "ProbabilityEstimate", "model_for_category", "REGISTRY"]
