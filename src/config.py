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


class RiskConfig(BaseModel):
    max_position_size_usd: float = 40
    max_concurrent_positions: int = 15
    max_open_exposure_pct: float = 0.40
    max_daily_loss_usd: float = 100
    max_consecutive_losses: int = 3
    cooldown_minutes_after_kill: int = 60


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
    weather: ModelToggle = Field(default_factory=lambda: ModelToggle(enabled=True))
    econ: ModelToggle = Field(default_factory=ModelToggle)
    sports: ModelToggle = Field(default_factory=ModelToggle)
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
