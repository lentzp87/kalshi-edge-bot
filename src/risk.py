"""Risk engine — the survival system.

Holds in-memory state about today's P&L, open exposure, and consecutive losses.
Every signal must `approve()` here before execution. Every closed trade calls
`record_close()` so the engine can react.

The kill switch trips when ANY of the following is true:
  - daily realized loss exceeds max_daily_loss_usd
  - consecutive losses >= max_consecutive_losses
  - open exposure already >= max_open_exposure_pct of bankroll

A tripped engine refuses all new signals until the cooldown elapses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog

from .config import file_config
from .decision import TradeSignal

log = structlog.get_logger(__name__)


@dataclass
class _RiskState:
    open_exposure_usd: float = 0.0
    open_positions: int = 0
    realized_pnl_today_usd: float = 0.0
    consecutive_losses: int = 0
    last_reset_date: str = ""
    killed_until: datetime | None = None


@dataclass
class RiskEngine:
    state: _RiskState = field(default_factory=_RiskState)

    def _reset_day_if_needed(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.state.last_reset_date != today:
            self.state.last_reset_date = today
            self.state.realized_pnl_today_usd = 0.0
            self.state.consecutive_losses = 0
            self.state.killed_until = None
            log.info("risk.daily_reset")

    def _is_killed(self) -> bool:
        if self.state.killed_until and datetime.now(timezone.utc) < self.state.killed_until:
            return True
        return False

    def _trip(self, reason: str) -> None:
        cfg = file_config().risk
        until = datetime.now(timezone.utc) + timedelta(minutes=cfg.cooldown_minutes_after_kill)
        self.state.killed_until = until
        log.warning("risk.kill_switch", reason=reason, cooldown_until=until.isoformat())

    def approve(self, signal: TradeSignal) -> bool:
        self._reset_day_if_needed()
        cfg = file_config()

        if self._is_killed():
            log.info("risk.reject.killed", ticker=signal.ticker)
            return False

        if signal.size_usd > cfg.risk.max_position_size_usd:
            log.info("risk.reject.size", ticker=signal.ticker, size=signal.size_usd)
            return False

        if self.state.open_positions >= cfg.risk.max_concurrent_positions:
            log.info("risk.reject.position_count")
            return False

        new_exposure = self.state.open_exposure_usd + signal.size_usd
        if new_exposure > cfg.bankroll_usd * cfg.risk.max_open_exposure_pct:
            log.info("risk.reject.exposure_cap")
            return False

        return True

    def record_open(self, signal: TradeSignal) -> None:
        self.state.open_exposure_usd += signal.size_usd
        self.state.open_positions += 1

    def record_close(self, *, size_usd: float, realized_pnl_usd: float) -> None:
        self.state.open_exposure_usd = max(0.0, self.state.open_exposure_usd - size_usd)
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.realized_pnl_today_usd += realized_pnl_usd

        if realized_pnl_usd < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        cfg = file_config().risk
        if self.state.realized_pnl_today_usd <= -cfg.max_daily_loss_usd:
            self._trip("daily_loss_cap")
        elif self.state.consecutive_losses >= cfg.max_consecutive_losses:
            self._trip("consecutive_losses")
