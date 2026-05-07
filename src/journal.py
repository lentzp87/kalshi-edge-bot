"""SQLite trade journal.

Two tables:
  signals — every model output, even the ones we didn't trade
  trades  — opened + closed positions with realized P&L

The dashboard reads from here. The realized-edge tracker on the dashboard
is what tells you whether your model is actually predictive.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog

from .config import env_config
from .decision import TradeSignal
from .kalshi_client import Market

log = structlog.get_logger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    ticker TEXT NOT NULL,
    category TEXT,
    market_p REAL,
    model_p REAL,
    edge REAL,
    confidence REAL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    contracts INTEGER NOT NULL,
    size_usd REAL NOT NULL,
    fill_price REAL NOT NULL,
    opened_ts TEXT NOT NULL,
    closed_ts TEXT,
    exit_price REAL,
    pnl_usd REAL,
    fees_usd REAL,
    exit_reason TEXT,
    edge REAL,
    reason TEXT
);
"""


class Journal:
    def __init__(self) -> None:
        env = env_config()
        Path(env.data_dir).mkdir(parents=True, exist_ok=True)
        # DB version log:
        #   trades.db          (v1) — pre-weather-pivot, abandoned
        #   trades_sports.db   (v2) — pre P&L sign-flip fix; data corrupted
        #                              by inverted NO-side P&L. Abandoned.
        #   trades_sports_v3.db — current. Started after the NO-side fix.
        # Old DBs remain on disk for archaeology but are no longer read.
        self.path = Path(env.data_dir) / "trades_sports_v3.db"
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_signal(self, market: Market, *, model_p: float, edge: float, confidence: float, reason: str) -> None:
        self.conn.execute(
            "INSERT INTO signals (ts, ticker, category, market_p, model_p, edge, confidence, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self._now(), market.ticker, market.category, market.mid, model_p, edge, confidence, reason),
        )
        self.conn.commit()

    def log_open(self, *, signal: TradeSignal, market: Market, fill_price: float, contracts: int) -> None:
        self.conn.execute(
            "INSERT INTO trades (ticker, side, contracts, size_usd, fill_price, opened_ts, edge, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                signal.ticker,
                signal.side,
                contracts,
                signal.size_usd,
                fill_price,
                self._now(),
                signal.edge,
                signal.reason,
            ),
        )
        self.conn.commit()

    def log_close(self, *, ticker: str, exit_price: float, pnl_usd: float, fees_usd: float, reason: str) -> None:
        self.conn.execute(
            "UPDATE trades SET closed_ts=?, exit_price=?, pnl_usd=?, fees_usd=?, exit_reason=? "
            "WHERE ticker=? AND closed_ts IS NULL",
            (self._now(), exit_price, pnl_usd, fees_usd, reason, ticker),
        )
        self.conn.commit()

    # ---------- read helpers used by dashboard ----------

    def open_positions(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT ticker, side, contracts, size_usd, fill_price, opened_ts, edge "
            "FROM trades WHERE closed_ts IS NULL ORDER BY opened_ts DESC"
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def recent_trades(self, limit: int = 100) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM trades ORDER BY opened_ts DESC LIMIT ?", (limit,)
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def daily_pnl(self) -> dict:
        cur = self.conn.execute(
            "SELECT DATE(closed_ts) AS d, SUM(pnl_usd) AS pnl, COUNT(*) AS n "
            "FROM trades WHERE closed_ts IS NOT NULL GROUP BY d ORDER BY d DESC LIMIT 30"
        )
        return {row[0]: {"pnl": row[1], "n": row[2]} for row in cur.fetchall()}

    def realized_edge_summary(self) -> dict:
        """Predicted edge vs realized P&L. Bucketed by edge band."""
        cur = self.conn.execute(
            "SELECT edge, pnl_usd, size_usd FROM trades WHERE closed_ts IS NOT NULL"
        )
        buckets: dict[str, dict[str, float]] = {}
        for edge, pnl, size in cur.fetchall():
            bucket = f"{round((edge or 0) * 100):>3}bp"
            b = buckets.setdefault(bucket, {"n": 0, "predicted": 0, "realized": 0})
            b["n"] += 1
            b["predicted"] += (edge or 0) * (size or 0)
            b["realized"] += pnl or 0
        return buckets
