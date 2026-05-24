"""Shadow models — alternative signal generators run OBSERVE-ONLY.

Each shadow model produces a pick for a market WITHOUT affecting live
trading. Picks are persisted to the journal's `shadow_signals` table so
every model can be backtested in isolation against Kalshi resolutions.

Three models registered (2026-05-24):

  baseline       Kalshi structural-bias play. Moderate favorites
                 (executable price 0.60-0.80) are underpriced per
                 Bürgi/Deng/Whelan (2026, 300k contracts). There is no
                 model here — the pick is simply "buy the favorite,
                 hold to settlement." This is the NULL HYPOTHESIS: any
                 real model must beat it or it isn't adding value.

  steam          Line-movement momentum. If a market's mid has moved
                 sharply and consistently over ~20 min, that move is
                 (probably) sharp money — bet WITH it. Distinct signal:
                 it keys off the price DELTA, not the level.

  cross_exchange If Kalshi disagrees with Polymarket on the same event
                 (from the cross_exchange spread snapshot), bet toward
                 the cross-exchange consensus.

Promotion path: observe ~2 weeks, backtest each model's picks vs
outcomes with a join script, promote only models that beat `baseline`.
Nothing here places a trade.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from .kalshi_client import Market

log = structlog.get_logger(__name__)


# ---- tuning ---------------------------------------------------------------
BASELINE_LO = 0.60          # structural-bias favorite price band
BASELINE_HI = 0.80
BASELINE_BUMP = 0.03        # assumed underpricing (fair = price + bump)

STEAM_WINDOW_SEC = 20 * 60  # compare mid now vs ~20 min ago
STEAM_RETENTION_SEC = 40 * 60
STEAM_MIN_MOVE = 0.04       # 4pp move to call it steam

CROSS_MIN_SPREAD = 0.04     # 4pp Kalshi-vs-Polymarket gap to act


@dataclass
class ShadowSignal:
    model: str          # "baseline" / "steam" / "cross_exchange"
    ticker: str
    side: str           # "yes" / "no"
    prob: float         # the model's fair probability for that side
    edge: float         # prob - executable price
    kalshi_price: float # executable price for the picked side
    note: str


# Per-ticker mid history for the steam model. Module-level so it
# survives across scan iterations.
_mid_history: dict[str, list[tuple[float, float]]] = {}


def _favorite_side(market: Market) -> tuple[str, float]:
    """Return (side, executable_price) for the FAVORITE — the side the
    market thinks is more likely. Executable price is what we'd pay."""
    mid = market.mid
    if mid >= 0.5:
        # YES is the favorite; we'd pay yes_ask to buy it.
        return "yes", market.effective_yes_ask
    # NO is the favorite; buying NO costs (1 - yes_bid).
    return "no", 1.0 - market.effective_yes_bid


def _baseline(market: Market) -> ShadowSignal | None:
    """Structural-bias baseline: moderate favorites are underpriced."""
    side, price = _favorite_side(market)
    if not (0 < price < 1):
        return None
    if not (BASELINE_LO <= price <= BASELINE_HI):
        return None
    fair = min(0.99, price + BASELINE_BUMP)
    return ShadowSignal(
        model="baseline", ticker=market.ticker, side=side,
        prob=fair, edge=fair - price, kalshi_price=price,
        note=f"favorite@{price:.2f}",
    )


def _steam(market: Market) -> ShadowSignal | None:
    """Line-movement momentum — bet WITH a sharp, consistent recent move."""
    now = time.time()
    mid = market.mid
    if not (0 < mid < 1):
        return None
    hist = _mid_history.setdefault(market.ticker, [])
    hist.append((now, mid))
    # Prune anything older than the retention window.
    cutoff = now - STEAM_RETENTION_SEC
    if hist and hist[0][0] < cutoff:
        hist[:] = [h for h in hist if h[0] >= cutoff]
    # Find the snapshot closest to (now - STEAM_WINDOW_SEC).
    target = now - STEAM_WINDOW_SEC
    ref = None
    best_dist = float("inf")
    for ts, m in hist:
        if ts > now - 60:       # skip the just-pushed point
            continue
        d = abs(ts - target)
        if d < best_dist:
            best_dist, ref = d, m
    if ref is None:
        return None
    delta = mid - ref           # +ve: YES side steaming up
    if abs(delta) < STEAM_MIN_MOVE:
        return None
    # Bet WITH the move. If YES steamed up, buy YES at yes_ask.
    if delta > 0:
        side, price = "yes", market.effective_yes_ask
    else:
        side, price = "no", 1.0 - market.effective_yes_bid
    if not (0 < price < 1):
        return None
    # Crude fair: assume the move continues ~half as far again.
    fair = min(0.99, max(0.01, price + abs(delta) * 0.5))
    return ShadowSignal(
        model="steam", ticker=market.ticker, side=side,
        prob=fair, edge=fair - price, kalshi_price=price,
        note=f"mid {ref:.2f}->{mid:.2f} ({delta:+.2f})",
    )


def _cross_exchange(market: Market) -> ShadowSignal | None:
    """Bet toward the Kalshi-vs-Polymarket consensus when they disagree."""
    try:
        from . import cross_exchange_state
        snap = cross_exchange_state.latest()
    except Exception:  # noqa: BLE001
        return None
    for s in snap.get("spreads", []):
        if s.get("kalshi_ticker") != market.ticker:
            continue
        # spread_pp is signed: kalshi_yes - polymarket_yes.
        spread = float(s.get("spread_pp") or 0.0)
        if abs(spread) < CROSS_MIN_SPREAD:
            return None
        poly_yes = float(s.get("polymarket_yes_price") or 0.0)
        if not (0 < poly_yes < 1):
            return None
        if spread < 0:
            # Kalshi YES cheaper than Polymarket -> buy Kalshi YES.
            side, price, fair = "yes", market.effective_yes_ask, poly_yes
        else:
            # Kalshi YES dearer -> Polymarket likes NO -> buy Kalshi NO.
            side = "no"
            price = 1.0 - market.effective_yes_bid
            fair = 1.0 - poly_yes
        if not (0 < price < 1):
            return None
        return ShadowSignal(
            model="cross_exchange", ticker=market.ticker, side=side,
            prob=fair, edge=fair - price, kalshi_price=price,
            note=f"kalshi-poly spread {spread:+.2f}",
        )
    return None


_MODELS = (_baseline, _steam, _cross_exchange)


def run_shadows(market: Market, journal) -> list[ShadowSignal]:
    """Run every shadow model on `market`. Persist + log each pick.
    Returns the list of signals produced (may be empty). Never raises
    — a shadow-model failure must not disturb the live trading loop.
    """
    out: list[ShadowSignal] = []
    for fn in _MODELS:
        try:
            sig = fn(market)
        except Exception:  # noqa: BLE001
            log.warning("shadow.model_error", model=fn.__name__,
                        ticker=market.ticker)
            continue
        if sig is None:
            continue
        out.append(sig)
        log.info("shadow.pick", model=sig.model, ticker=sig.ticker,
                 side=sig.side, prob=round(sig.prob, 3),
                 edge=round(sig.edge, 3), price=round(sig.kalshi_price, 3))
        try:
            journal.log_shadow_signal(
                model=sig.model, ticker=sig.ticker, side=sig.side,
                prob=sig.prob, edge=sig.edge,
                kalshi_price=sig.kalshi_price, note=sig.note,
            )
        except Exception:  # noqa: BLE001
            log.warning("shadow.persist_error", model=sig.model,
                        ticker=sig.ticker)
    return out
