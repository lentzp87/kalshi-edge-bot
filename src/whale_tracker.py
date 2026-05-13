"""Whale-activity detector for Kalshi markets.

Kalshi doesn't stream individual trades, so we can't see a $1k bet
hit the tape directly. Instead we infer whale activity from deltas
between polls:

  * Price jump: last_price moves >= 5c since previous poll. A move
    that big requires real size to clear the book.
  * Volume burst: volume_24h jumps by >= 5000 contracts (~$2-3k of
    notional value) between polls. Could be a single whale or a
    coincident pile of small bets — we treat both the same.
  * Resting whale: top-of-book size >= 2000 contracts. The order may
    be spoofed, but real money sometimes parks size to anchor a
    price.

A market is "whale active" if any of these triggered in the last
WINDOW_MIN minutes. The decision layer can then boost size when the
whale's direction aligns with our model's chosen side.

Trade-offs vs streaming
-----------------------
Snapshot polls give us a 30-second lag at best. Real sharps act on
seconds. So we'll usually catch the whale AFTER they've moved the
price — meaning the edge they had is already gone, and we'd just
be retail-chasing. Two ways this still produces value:

  1. The whale's bet is a CONFIRMATION signal, not a follow signal.
     If our model already says "buy YES" and a whale just bought YES,
     the convergence is meaningful.
  2. Sometimes whales build positions over many minutes / hours.
     Catching the second or third clip is fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from .kalshi_client import Market

log = structlog.get_logger(__name__)


# Tuneable thresholds. Conservative defaults.
PRICE_JUMP_THRESHOLD = 0.05        # 5c move = potential whale
VOLUME_BURST_THRESHOLD = 5000      # 5000 contracts ~ $2-3k notional
RESTING_SIZE_THRESHOLD = 2000      # 2000 contracts at top of book
SIGNAL_WINDOW_SECONDS = 15 * 60    # remember whale signals for 15 min

# Baseline-comparison window. We diff the latest snapshot against the
# snapshot closest to (now - BASELINE_WINDOW_SECONDS). This makes the
# detector source-agnostic — slow 30s scanner polls and high-rate
# WebSocket ticks both work because we always compare against a
# ~10s-old reference, not just "the previous call to update()".
BASELINE_WINDOW_SECONDS = 10
HISTORY_RETENTION_SECONDS = 60     # drop snapshots older than this


@dataclass
class _Snapshot:
    ts: float
    last_price: float
    volume_24h: float
    yes_bid_size: float
    yes_ask_size: float
    # NO-side book sizes. Kalshi exposes these separately from the
    # yes_* sizes. A large NO bid = whale BUYING NO directly (bearish
    # on YES), a large NO ask = whale SELLING NO (bullish on YES via
    # the arb relationship NO_ask = 1 - YES_bid). We were missing
    # both before because we only watched the yes_* half.
    no_bid_size: float = 0.0
    no_ask_size: float = 0.0


@dataclass
class WhaleSignal:
    ticker: str
    detected_at: float
    direction: str       # "yes", "no", or "?" (unknown side)
    confidence: float    # 0.0-1.0 — how strong the signal is
    reason: str          # human-readable: "price_jump_+7c", etc.


class WhaleTracker:
    """Per-process tracker. Holds previous snapshots in memory; on
    each market update, compares to previous snapshot and emits a
    signal if a whale-shaped delta appears.
    """

    def __init__(self) -> None:
        # Per-ticker bounded history of snapshots, newest at the end. We
        # diff "new" against the snapshot closest to (now - BASELINE_WINDOW).
        # That makes WS ticks and scanner polls both produce correct deltas
        # — the comparison reference is time-bounded, not call-bounded.
        self._history: dict[str, list[_Snapshot]] = {}
        self._signals: dict[str, list[WhaleSignal]] = {}

    @staticmethod
    def _select_baseline(history: list[_Snapshot], now_ts: float) -> _Snapshot | None:
        """Pick the snapshot closest to (now_ts - BASELINE_WINDOW_SECONDS).
        Returns None if there's no snapshot at least 1s older than now_ts."""
        target = now_ts - BASELINE_WINDOW_SECONDS
        best = None
        best_dist = float("inf")
        for s in history:
            if s.ts >= now_ts - 0.5:
                continue  # skip the just-pushed snapshot
            dist = abs(s.ts - target)
            if dist < best_dist:
                best_dist = dist
                best = s
        return best

    def update(self, market: Market) -> WhaleSignal | None:
        """Feed a fresh market snapshot. Returns a WhaleSignal if a
        whale-shaped delta was detected vs the ~10s-old baseline.

        Source-agnostic: scanner polls (30s gap) and WS ticks (many per
        second) both work because we diff against the time-targeted
        baseline, not the immediately-previous call.
        """
        now = time.time()
        def _size(key: str) -> float:
            try:
                return float(market.raw.get(key) or 0)
            except (TypeError, ValueError):
                return 0.0
        yes_bid_size = _size("yes_bid_size_fp")
        yes_ask_size = _size("yes_ask_size_fp")
        no_bid_size  = _size("no_bid_size_fp")
        no_ask_size  = _size("no_ask_size_fp")
        new = _Snapshot(
            ts=now,
            last_price=float(market.last_price or 0),
            volume_24h=float(market.volume or 0),
            yes_bid_size=yes_bid_size,
            yes_ask_size=yes_ask_size,
            no_bid_size=no_bid_size,
            no_ask_size=no_ask_size,
        )
        history = self._history.setdefault(market.ticker, [])
        history.append(new)
        # Prune snapshots older than HISTORY_RETENTION_SECONDS so memory
        # doesn't grow unbounded under WS firehose.
        cutoff = now - HISTORY_RETENTION_SECONDS
        if history[0].ts < cutoff:
            history[:] = [s for s in history if s.ts >= cutoff]
        old = self._select_baseline(history, now)

        if not old:
            return None  # not enough history yet — nothing to compare

        signal: WhaleSignal | None = None

        # 1. Price jump — strongest signal
        price_delta = new.last_price - old.last_price
        if abs(price_delta) >= PRICE_JUMP_THRESHOLD and new.last_price > 0:
            direction = "yes" if price_delta > 0 else "no"
            confidence = min(abs(price_delta) / 0.10, 1.0)  # 10c = max conf
            sign = "+" if price_delta > 0 else ""
            signal = WhaleSignal(
                ticker=market.ticker, detected_at=now,
                direction=direction, confidence=confidence,
                reason=f"price_jump_{sign}{int(price_delta * 100)}c",
            )

        # 2. Volume burst — medium signal (don't override price jump)
        elif (new.volume_24h - old.volume_24h) >= VOLUME_BURST_THRESHOLD:
            vol_delta = new.volume_24h - old.volume_24h
            # Direction is ambiguous from volume alone; tag it "?"
            signal = WhaleSignal(
                ticker=market.ticker, detected_at=now,
                direction="?", confidence=0.5,
                reason=f"volume_burst_{int(vol_delta)}",
            )

        # 3. Resting whale — weakest. Watch BOTH halves of the book:
        #   large_yes_bid / large_no_ask  → bullish on YES
        #   large_yes_ask / large_no_bid  → bullish on NO
        # No-side detection added 2026-05-13. Previously we only saw
        # YES-book resting orders; the NO-book half was invisible to us
        # (a whale parking a $5k NO bid would only show up in our data
        # after the YES book re-priced via arb, ~30s late).
        elif (yes_bid_size >= RESTING_SIZE_THRESHOLD
              and (old.yes_bid_size or 0) < RESTING_SIZE_THRESHOLD):
            signal = WhaleSignal(
                ticker=market.ticker, detected_at=now,
                direction="yes", confidence=0.4,
                reason=f"large_yes_bid_{int(yes_bid_size)}c",
            )
        elif (no_ask_size >= RESTING_SIZE_THRESHOLD
              and (old.no_ask_size or 0) < RESTING_SIZE_THRESHOLD):
            # whale selling NO = covering / closing NO = bullish on YES
            signal = WhaleSignal(
                ticker=market.ticker, detected_at=now,
                direction="yes", confidence=0.4,
                reason=f"large_no_ask_{int(no_ask_size)}c",
            )
        elif (yes_ask_size >= RESTING_SIZE_THRESHOLD
              and (old.yes_ask_size or 0) < RESTING_SIZE_THRESHOLD):
            signal = WhaleSignal(
                ticker=market.ticker, detected_at=now,
                direction="no", confidence=0.4,
                reason=f"large_yes_ask_{int(yes_ask_size)}c",
            )
        elif (no_bid_size >= RESTING_SIZE_THRESHOLD
              and (old.no_bid_size or 0) < RESTING_SIZE_THRESHOLD):
            # whale BUYING NO directly = bearish on YES
            signal = WhaleSignal(
                ticker=market.ticker, detected_at=now,
                direction="no", confidence=0.4,
                reason=f"large_no_bid_{int(no_bid_size)}c",
            )

        if signal:
            self._signals.setdefault(market.ticker, []).append(signal)
            # Trim old signals
            cutoff = now - SIGNAL_WINDOW_SECONDS
            self._signals[market.ticker] = [
                s for s in self._signals[market.ticker]
                if s.detected_at >= cutoff
            ]
            log.info("whale.detected",
                     ticker=market.ticker, direction=signal.direction,
                     confidence=round(signal.confidence, 2),
                     reason=signal.reason)
        return signal

    def recent_signals(self, ticker: str) -> list[WhaleSignal]:
        """All whale signals for this ticker within the last 15 min."""
        cutoff = time.time() - SIGNAL_WINDOW_SECONDS
        return [s for s in self._signals.get(ticker, []) if s.detected_at >= cutoff]

    def has_aligned_signal(self, ticker: str, our_side: str) -> WhaleSignal | None:
        """Return the most recent signal whose direction matches our_side
        (or is unknown), if any. Used by the decision layer to confirm/boost.
        """
        signals = self.recent_signals(ticker)
        if not signals:
            return None
        # Prefer a directional signal that matches; else accept "?" (volume burst)
        for s in reversed(signals):  # most recent first
            if s.direction == our_side:
                return s
        for s in reversed(signals):
            if s.direction == "?":
                return s
        return None
