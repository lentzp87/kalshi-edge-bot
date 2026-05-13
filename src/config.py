"""Centralized config: env vars + config.yaml merged into typed objects."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScannerConfig(BaseModel):
    loop_interval_seconds: int = 30
    max_pages_per_scan: int = 50
    price_min: float = 0.20
    price_max: float = 0.80
    max_spread_cents: int = 4
    min_liquidity_usd: float = 200
    max_minutes_to_expiry: int = 4320
    min_minutes_to_expiry: int = 30
    series_tickers: list[str] = Field(default_factory=list)
    allowed_categories: list[str] = Field(default_factory=list)


class DecisionConfig(BaseModel):
    min_edge: float = 0.06
    kelly_fraction: float = 0.25
    # Entry price floor. `min_entry_price` is the legacy single knob.
    # `min_entry_price_yes` / `min_entry_price_no` allow side-specific
    # floors — in a pure two-way market YES at $0.30 ≡ NO at $0.70,
    # so symmetric thresholds are the right default. We keep the knobs
    # separate to allow asymmetric tuning once we have N>=50 in the
    # dashboard's by_side_x_entry panel. If either side-specific value
    # is None, the legacy `min_entry_price` is used for that side.
    min_entry_price: float = 0.55
    min_entry_price_yes: float | None = None
    min_entry_price_no: float | None = None
    # Minimum model probability for OUR chosen side. If model thinks our
    # side wins 60%+, fire. If it's a coin flip or worse, skip — even
    # if the edge looks positive, the variance/fee math is unfriendly.
    # Setting 0.0 disables this filter (back-compat default).
    min_p_yes: float = 0.0
    # Safety margin (in probability points) added to the per-market
    # required edge after entry+exit fees + half-spread + slippage.
    # Don't fire on knife-edge gross edges that barely cover costs.
    required_edge_safety_pp: float = 0.005


class RiskConfig(BaseModel):
    max_position_size_usd: float = 40
    max_concurrent_positions: int = 15
    max_open_exposure_pct: float = 0.40
    max_daily_loss_usd: float = 100
    max_consecutive_losses: int = 3
    cooldown_minutes_after_kill: int = 60
    # Larger cap when an aligned whale signal exists for the ticker
    # (price jumped in our direction, big volume burst, etc.). Set
    # equal to max_position_size_usd to disable the boost.
    whale_max_position_size_usd: float = 40


class ExecutionConfig(BaseModel):
    take_profit_pct: float = 0.20
    stop_loss_pct: float = 0.12
    time_exit_minutes: int = 90
    order_type: Literal["limit", "market"] = "limit"
    scale_in_chunks: int = 2


class ModelToggle(BaseModel):
    enabled: bool = False
    # arbitrary per-model knobs survive here
    extra: dict = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    sports: ModelToggle = Field(default_factory=ModelToggle)
    econ: ModelToggle = Field(default_factory=ModelToggle)
    crypto: ModelToggle = Field(default_factory=ModelToggle)
    politics: ModelToggle = Field(default_factory=ModelToggle)


class FileConfig(BaseModel):
    bankroll_usd: float = 2000
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)


class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kalshi_api_key_id: str = "missing"
    kalshi_api_private_key_path: str | None = None
    kalshi_api_private_key: str | None = None  # PEM contents, optional alternative
    kalshi_env: Literal["demo", "prod"] = "demo"
    mode: Literal["paper", "live"] = "paper"

    open_meteo_base: str = "https://api.open-meteo.com/v1"
    nws_user_agent: str = "kalshi-edge-bot"

    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000

    data_dir: str = "./data"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache
def file_config() -> FileConfig:
    raw = _load_yaml(Path("config.yaml"))

    # Allow per-model `extra` to be specified as flat keys in YAML.
    models_raw = raw.get("models", {})
    for name, cfg in list(models_raw.items()):
        if isinstance(cfg, dict):
            enabled = cfg.pop("enabled", False)
            models_raw[name] = {"enabled": enabled, "extra": cfg}
    return FileConfig.model_validate(raw)


@lru_cache
def env_config() -> EnvConfig:
    return EnvConfig()
