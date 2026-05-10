"""Shared in-memory snapshot of the latest Kalshi-vs-Polymarket spreads.

The trading-loop process owns the writer (a periodic task in main.py),
the FastAPI dashboard reads from it. A single module-level dict + a
non-async `latest()` accessor is enough — both producers and consumers
live in the same process.

Schema:
    {
        "ts": "2026-05-10T12:34:56+00:00",   # ISO UTC of the snapshot
        "n_kalshi": 1234,                    # markets considered
        "n_polymarket": 9876,
        "spreads": [ <CrossExchangeSpread.__dict__> ... ]   # already
                                                              top_n filtered
    }
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

# Module-level snapshot. Always a dict (never None) so the dashboard
# doesn't have to special-case the cold-start window.
_snapshot: dict[str, Any] = {
    "ts": None,
    "n_kalshi": 0,
    "n_polymarket": 0,
    "spreads": [],
}


def update(*, n_kalshi: int, n_polymarket: int, spreads: list) -> None:
    """Replace the current snapshot. `spreads` is a list of
    CrossExchangeSpread dataclass instances; we serialize via asdict so
    the dashboard JSON path is straightforward.
    """
    serialized: list[dict] = []
    for s in spreads:
        try:
            serialized.append(asdict(s))
        except TypeError:
            # Defensive: tolerate already-dict entries.
            if isinstance(s, dict):
                serialized.append(s)
    _snapshot["ts"] = datetime.now(timezone.utc).isoformat()
    _snapshot["n_kalshi"] = n_kalshi
    _snapshot["n_polymarket"] = n_polymarket
    _snapshot["spreads"] = serialized


def latest() -> dict[str, Any]:
    """Read the current snapshot. Returns a shallow copy so callers can
    mutate freely without corrupting the shared state.
    """
    return {
        "ts": _snapshot["ts"],
        "n_kalshi": _snapshot["n_kalshi"],
        "n_polymarket": _snapshot["n_polymarket"],
        "spreads": list(_snapshot["spreads"]),
    }
