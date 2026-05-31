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
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

from .config import env_config
from .decision import TradeSignal
from .kalshi_client import Market

log = structlog.get_logger(__name__)

# Dedup window for log_shadow_signal. The same (model, ticker, side)
# can only re-log this often. 1h is plenty — settled_outcome is what we
# care about, not how many times the same pick was emitted intra-hour.
SHADOW_DEDUP_SECONDS = 3600
# Prune entries from the dedup cache every N writes so it can't grow
# unbounded across a long-running process.
SHADOW_DEDUP_PRUNE_EVERY = 500


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
    reason TEXT,
    -- Kalshi mid price ~5 min before tipoff, captured even if we
    -- already exited via TP/SL. Lets us measure CLV (closing-line
    -- value) against our own fill price without needing an external
    -- book API. CLV = clv_price - fill_price (positive if line moved
    -- our way regardless of side, since `fill_price` and `clv_price`
    -- are both stored in the side we bought).
    clv_price REAL
);

-- Observe-only picks from src/shadow_models.py. Every alternative
-- model logs its pick here so each can be backtested in isolation
-- against Kalshi resolutions. Nothing here is a real position.
CREATE TABLE IF NOT EXISTS shadow_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    prob REAL,
    edge REAL,
    kalshi_price REAL,
    note TEXT,
    -- filled later by the settlement backfill: 1 if the picked side
    -- won, 0 if it lost, NULL until the Kalshi market resolves.
    settled_outcome INTEGER,
    settled_checked_ts TEXT
);
"""


# Schema migration for existing v3 DBs that pre-date newer columns.
# Idempotent: try to add the column, ignore the error if it already exists.
_MIGRATIONS = [
    "ALTER TABLE trades ADD COLUMN clv_price REAL",
    # Settlement backtest columns: lets us compute "what if we'd held to
    # game resolution instead of stop_loss/take_profit". settled_outcome
    # is 1 if our side won, 0 if it lost, NULL if not yet resolved.
    # settlement_pnl_usd is the counterfactual P&L net of entry fee.
    "ALTER TABLE trades ADD COLUMN settled_outcome INTEGER",
    "ALTER TABLE trades ADD COLUMN settlement_pnl_usd REAL",
    "ALTER TABLE trades ADD COLUMN settled_checked_ts TEXT",
    # Live mark-to-market: watcher writes the current mid every 15s for
    # each open position so the dashboard can show current P&L without
    # having to make Kalshi API calls itself.
    "ALTER TABLE trades ADD COLUMN current_mid REAL",
    "ALTER TABLE trades ADD COLUMN current_mid_ts TEXT",
    # Per-position mid range during the entire hold. Lets the dashboard
    # compute retrospective "would TP only have fired" / "would SL only
    # have fired" without needing a full price-history table.
    "ALTER TABLE trades ADD COLUMN max_mid_during_hold REAL",
    "ALTER TABLE trades ADD COLUMN min_mid_during_hold REAL",
    # Mid price snapshotted once when a position crosses ~75 min old.
    # Lets the A/B exit simulator backtest a "hard exit at 75 min"
    # policy without a full price-history table. Instrument-only —
    # the live exit logic is unchanged. NULL for trades that closed
    # before 75 min or that opened before this column existed.
    "ALTER TABLE trades ADD COLUMN mid_at_75min REAL",
]


class Journal:
    def __init__(self) -> None:
        env = env_config()
        Path(env.data_dir).mkdir(parents=True, exist_ok=True)
        # DB version log:
        #   trades.db          (v1) — pre-weather-pivot, abandoned
        #   trades_sports.db   (v2) — pre P&L sign-flip fix; data corrupted
        #                              by inverted NO-side P&L. Abandoned.
        #   trades_sports_v3.db — pre-whale + pre-min_p_yes filter; mixed
        #                              data from many strategy iterations.
        #   trades_sports_v4.db — current. Clean slate to evaluate the
        #                              new (min_p_yes=0.60, whale boost,
        #                              tipoff-aligned exits) ruleset.
        # Old DBs remain on disk for archaeology but are no longer read.
        self.path = Path(env.data_dir) / "trades_sports_v4.db"
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        # Apply additive migrations idempotently — SQLite ALTER fails if
        # the column already exists; we swallow that and move on.
        for stmt in _MIGRATIONS:
            try:
                self.conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        self.conn.commit()
        # In-memory dedup cache for log_shadow_signal. Maps
        # (model, ticker, side) -> last-logged unix ts. We skip the
        # INSERT if the same key was logged within SHADOW_DEDUP_SECONDS.
        # Without this the table grows ~150k rows/day (every model on
        # every market on every scan). The cache itself is pruned
        # opportunistically every PRUNE_EVERY inserts so it can't blow
        # memory either.
        self._shadow_dedup: dict[tuple[str, str, str], float] = {}
        self._shadow_dedup_writes_since_prune = 0

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

    def log_open(self, *, signal: TradeSignal, market: Market, fill_price: float,
                 contracts: int, opened_ts: str | None = None) -> None:
        """Insert an opened-position row.

        `opened_ts` MUST be the caller's `pos.opened_at.isoformat()`, not a
        fresh timestamp. The CLV sampler and the mark-to-market watcher both
        issue `UPDATE ... WHERE ticker=? AND opened_ts=?` using the in-memory
        OpenPosition's `opened_at`. If log_open stamps its own `_now()` here,
        that value never matches the watchers' value and every CLV / mid-range
        UPDATE silently affects zero rows. (Root cause of n_clv=0 and the
        degenerate A/B exit panel — fixed 2026-05-21.)
        """
        self.conn.execute(
            "INSERT INTO trades (ticker, side, contracts, size_usd, fill_price, opened_ts, edge, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                signal.ticker,
                signal.side,
                contracts,
                signal.size_usd,
                fill_price,
                opened_ts or self._now(),
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

    def trades_pending_settlement(self, limit: int = 500) -> list[dict]:
        """Closed trades whose Kalshi market hasn't been resolved yet
        (or hasn't been checked). Used by the settlement backfill task."""
        cur = self.conn.execute(
            "SELECT * FROM trades "
            "WHERE closed_ts IS NOT NULL AND settled_outcome IS NULL "
            "ORDER BY closed_ts DESC LIMIT ?",
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_settlement(
        self, *, ticker: str, opened_ts: str,
        settled_outcome: int, settlement_pnl_usd: float,
    ) -> None:
        """Record the counterfactual 'held to settlement' P&L."""
        self.conn.execute(
            "UPDATE trades "
            "SET settled_outcome=?, settlement_pnl_usd=?, settled_checked_ts=? "
            "WHERE ticker=? AND opened_ts=?",
            (settled_outcome, settlement_pnl_usd, self._now(), ticker, opened_ts),
        )
        self.conn.commit()

    def settlement_summary(self) -> dict:
        """Aggregate stats: actual P&L vs held-to-settlement P&L."""
        cur = self.conn.execute(
            "SELECT pnl_usd, settlement_pnl_usd, exit_reason FROM trades "
            "WHERE closed_ts IS NOT NULL AND settled_outcome IS NOT NULL"
        )
        n = 0
        actual_total = 0.0
        settle_total = 0.0
        better_held = 0  # n trades where holding would have beaten exit
        worse_held = 0   # n trades where exit was better than holding
        sl_actual = 0.0
        sl_settle = 0.0
        sl_n = 0
        for actual, settle, reason in cur.fetchall():
            if actual is None or settle is None:
                continue
            n += 1
            actual_total += actual
            settle_total += settle
            if settle > actual:
                better_held += 1
            elif settle < actual:
                worse_held += 1
            if reason == "stop_loss":
                sl_n += 1
                sl_actual += actual
                sl_settle += settle
        return {
            "n_resolved": n,
            "actual_total": round(actual_total, 2),
            "settlement_total": round(settle_total, 2),
            "delta": round(settle_total - actual_total, 2),
            "better_held_pct": round(100 * better_held / n, 1) if n else 0.0,
            "stop_loss_n": sl_n,
            "stop_loss_actual": round(sl_actual, 2),
            "stop_loss_settlement": round(sl_settle, 2),
        }

    def update_current_mid(self, *, ticker: str, opened_ts: str, mid: float) -> None:
        """Record the latest mark-to-market mid for an open position.
        Also rolls max_mid_during_hold and min_mid_during_hold so the
        dashboard can answer 'would TP-only have fired?' / 'would
        SL-only have fired?' without a separate snapshots table.
        """
        self.conn.execute(
            "UPDATE trades SET "
            "  current_mid=?, current_mid_ts=?, "
            "  max_mid_during_hold = MAX(COALESCE(max_mid_during_hold, ?), ?), "
            "  min_mid_during_hold = MIN(COALESCE(min_mid_during_hold, ?), ?) "
            "WHERE ticker=? AND opened_ts=? AND closed_ts IS NULL",
            (mid, self._now(), mid, mid, mid, mid, ticker, opened_ts),
        )
        self.conn.commit()

    def log_shadow_signal(self, *, model: str, ticker: str, side: str,
                          prob: float, edge: float, kalshi_price: float,
                          note: str) -> None:
        """Record one observe-only shadow-model pick. Backtested later
        by joining settled_outcome (filled by the settlement backfill).

        Deduped by (model, ticker, side): the same pick within
        SHADOW_DEDUP_SECONDS is a no-op. Without this the scanner re-logs
        identical picks every loop iteration — burned 887k rows in 6 days
        before this was added (2026-05-30 recovery). The win-rate calc
        only cares about unique picks that settle, so the extra rows
        had no analytical value, just bloat.
        """
        now_ts = time.time()
        key = (model, ticker, side)
        last_ts = self._shadow_dedup.get(key)
        if last_ts is not None and (now_ts - last_ts) < SHADOW_DEDUP_SECONDS:
            return
        self.conn.execute(
            "INSERT INTO shadow_signals "
            "(ts, model, ticker, side, prob, edge, kalshi_price, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self._now(), model, ticker, side, prob, edge, kalshi_price, note),
        )
        self.conn.commit()
        self._shadow_dedup[key] = now_ts
        # Opportunistic prune so the cache itself can't grow unbounded
        # across a long-running process.
        self._shadow_dedup_writes_since_prune += 1
        if self._shadow_dedup_writes_since_prune >= SHADOW_DEDUP_PRUNE_EVERY:
            cutoff = now_ts - SHADOW_DEDUP_SECONDS
            self._shadow_dedup = {
                k: ts for k, ts in self._shadow_dedup.items() if ts >= cutoff
            }
            self._shadow_dedup_writes_since_prune = 0

    def shadow_summary(self) -> list[dict]:
        """Per-model shadow-signal counts + resolved-pick win rate.
        Used by the dashboard's shadow-models panel."""
        cur = self.conn.execute(
            "SELECT model, COUNT(*) AS n, "
            "  SUM(CASE WHEN settled_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved, "
            "  SUM(CASE WHEN settled_outcome = 1 THEN 1 ELSE 0 END) AS wins, "
            "  AVG(edge) AS avg_edge "
            "FROM shadow_signals GROUP BY model ORDER BY n DESC"
        )
        out = []
        for model, n, resolved, wins, avg_edge in cur.fetchall():
            resolved = resolved or 0
            wins = wins or 0
            out.append({
                "model": model,
                "n": n,
                "resolved": resolved,
                "wins": wins,
                "win_rate": round(wins / resolved, 3) if resolved else 0.0,
                "avg_edge": round(avg_edge or 0.0, 4),
            })
        return out

    def shadow_signals_pending_settlement(self, limit: int = 1000) -> list[dict]:
        """Shadow picks not yet resolved against a Kalshi outcome."""
        cur = self.conn.execute(
            "SELECT * FROM shadow_signals WHERE settled_outcome IS NULL "
            "ORDER BY ts DESC LIMIT ?", (limit,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_shadow_settlement(self, *, signal_id: int,
                                 settled_outcome: int) -> None:
        """Record whether a shadow pick's side won (1) or lost (0)."""
        self.conn.execute(
            "UPDATE shadow_signals SET settled_outcome=?, settled_checked_ts=? "
            "WHERE id=?",
            (settled_outcome, self._now(), signal_id),
        )
        self.conn.commit()

    def cleanup_old_rows(self, *, shadow_max_age_days: int = 14) -> dict:
        """Prune the write-only signals table + stale unresolved shadow
        picks. Returns a dict of {table -> rows_deleted} for logging.

        - `signals` is write-only (nothing reads from it). Truncated
          every run.
        - `shadow_signals` resolved picks are KEPT forever — that's the
          backtest. Only unresolved picks older than
          `shadow_max_age_days` are dropped: those games are done and
          won't ever settle.
        - VACUUM at the end reclaims the freed disk pages. Briefly
          locks the DB; at our size (<50MB normal) this is <1s.

        Called from src/main.py's journal_cleanup_loop every 7 days.
        Without this, scanner-side write volume swelled the DB to
        492MB on 2026-05-30 and pushed the bot over its Render memory
        cap, causing dashboard reads to silently fail.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM signals")
        signals_before = int(cur.fetchone()[0])
        cur.execute("DELETE FROM signals")
        cur.execute(
            "DELETE FROM shadow_signals "
            "WHERE settled_outcome IS NULL "
            "  AND ts < datetime('now', ?)",
            (f"-{int(shadow_max_age_days)} days",),
        )
        shadow_unresolved_deleted = int(cur.rowcount)
        self.conn.commit()
        # VACUUM must run outside any transaction. The commit above
        # closes the implicit one started by DELETE.
        cur.execute("VACUUM")
        return {
            "signals_deleted": signals_before,
            "shadow_unresolved_deleted": shadow_unresolved_deleted,
        }

    def update_mid_at_75min(self, *, ticker: str, opened_ts: str, mid: float) -> None:
        """Snapshot the mid once, when a position crosses ~75 min old.
        The `mid_at_75min IS NULL` guard makes this write-once even if
        the watcher calls it on more than one tick past the threshold.
        """
        self.conn.execute(
            "UPDATE trades SET mid_at_75min=? "
            "WHERE ticker=? AND opened_ts=? AND mid_at_75min IS NULL",
            (mid, ticker, opened_ts),
        )
        self.conn.commit()

    def update_clv_price(self, *, ticker: str, opened_ts: str, clv_price: float) -> None:
        """Record the Kalshi mid price near tipoff for CLV measurement.

        Matches by (ticker, opened_ts) so we update the right row even
        when the same ticker has been traded multiple times across the
        session (rare with the executor's event lock + cooldown).
        """
        self.conn.execute(
            "UPDATE trades SET clv_price=? WHERE ticker=? AND opened_ts=?",
            (clv_price, ticker, opened_ts),
        )
        self.conn.commit()

    # ---------- read helpers used by dashboard ----------

    def open_positions(self) -> list[dict]:
        # Include `reason` so the dashboard can show team names, game,
        # provider, etc. Parsing happens client-side via parseReason().
        # current_mid is updated by the watcher every 15s, lets the
        # dashboard mark-to-market without making its own Kalshi calls.
        cur = self.conn.execute(
            "SELECT ticker, side, contracts, size_usd, fill_price, opened_ts, "
            "edge, reason, current_mid, current_mid_ts "
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
