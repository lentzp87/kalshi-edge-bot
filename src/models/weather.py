"""Weather temperature model.

Strategy
--------
Kalshi runs daily temperature markets for ~16 US cities (KXHIGHTDAL, KXLOWNY, ...)
with three market shapes per day:

    "95° or above"        - threshold ABOVE
    "86° or below"        - threshold BELOW
    "93° to 94°"          - bucket BETWEEN

We hit the free Open-Meteo Ensemble API (31-member GFS ensemble) for the
relevant city + date and compute the empirical CDF directly:

    P(YES) = (# ensemble members satisfying the market condition) / (total members)

This is what a published profitable Kalshi weather bot does, and it's what
we want: forecasts on temperature are a fundamental signal that the market
doesn't always reprice quickly. The ensemble agreement gives us a confidence
score for free.

External data: Open-Meteo Ensemble API
    https://ensemble-api.open-meteo.com/v1/ensemble
    No API key required. Rate limits are generous for daily forecasts.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import httpx
import structlog

from ..config import file_config
from ..kalshi_client import Market
from .base import ProbabilityEstimate

log = structlog.get_logger(__name__)


# Map a Kalshi weather series_ticker to (lat, lon, friendly_name).
# Kalshi uses two naming conventions: KXHIGH<CITY> (older) and KXHIGHT<CITY>
# (newer). We collapse both. Keys are checked as substrings of the ticker
# prefix (everything before the first '-' in the market ticker).
CITY_BY_SERIES: dict[str, tuple[float, float, str]] = {
    # New York
    "NY":     (40.7128,  -74.0060, "New York"),
    "NYC":    (40.7128,  -74.0060, "New York"),
    # Major US cities
    "DAL":    (32.7767,  -96.7970, "Dallas"),
    "AUS":    (30.2672,  -97.7431, "Austin"),
    "HOU":    (29.7604,  -95.3698, "Houston"),
    "SATX":   (29.4241,  -98.4936, "San Antonio"),
    "OKC":    (35.4676,  -97.5164, "Oklahoma City"),
    "NOLA":   (29.9511,  -90.0715, "New Orleans"),
    "ATL":    (33.7490,  -84.3880, "Atlanta"),
    "MIA":    (25.7617,  -80.1918, "Miami"),
    "CHI":    (41.8781,  -87.6298, "Chicago"),
    "DEN":    (39.7392, -104.9903, "Denver"),
    "PHX":    (33.4484, -112.0740, "Phoenix"),
    "LV":     (36.1716, -115.1391, "Las Vegas"),
    "LAX":    (34.0522, -118.2437, "Los Angeles"),
    "PHIL":   (39.9526,  -75.1652, "Philadelphia"),
    "DC":     (38.9072,  -77.0369, "Washington DC"),
    "SEA":    (47.6062, -122.3321, "Seattle"),
    "DV":     (36.5054, -117.0794, "Death Valley"),
}


def _city_from_series_ticker(series_ticker: str) -> tuple[float, float, str] | None:
    """Pull the city code off a Kalshi weather series ticker.

    Examples:
        KXHIGHTDAL  -> ("DAL", Dallas)
        KXLOWNY     -> ("NY", New York)
        KXLOWTLAX   -> ("LAX", Los Angeles)
        KXDVHIGH    -> ("DV", Death Valley)
        RAINSEA     -> not handled here (rain, not temp)
    """
    s = series_ticker.upper()
    # Try longest city codes first so SATX doesn't get mis-matched to AT
    for code in sorted(CITY_BY_SERIES.keys(), key=len, reverse=True):
        # Code must appear as the suffix or surrounded by letters that are
        # part of the prefix scheme (KXHIGHT, KXLOW, KXHIGH, KXLOWT, etc.)
        if s.endswith(code):
            return CITY_BY_SERIES[code]
    return None


def _is_high_market(series_ticker: str) -> bool:
    """Return True if this series is for daily HIGH temp, False for LOW."""
    s = series_ticker.upper()
    # 'LOW' is more specific — check first to avoid HIGH-prefix false positives
    if "LOW" in s:
        return False
    if "HIGH" in s:
        return True
    # Fallback: assume high
    return True


@dataclass
class _Strike:
    """What does a market resolve YES on?

    kind = "above"  -> YES if temp > threshold
    kind = "below"  -> YES if temp < threshold
    kind = "between" -> YES if low <= temp <= high
    """
    kind: str
    threshold: float = 0.0
    low: float = 0.0
    high: float = 0.0


def _parse_strike_from_yes_subtitle(yes_sub_title: str) -> _Strike | None:
    """Kalshi populates yes_sub_title with one of three patterns:

        "95° or above"
        "86° or below"
        "93° to 94°"

    We pull the numbers and the direction. Returns None if the string
    doesn't look like one of these.
    """
    if not yes_sub_title:
        return None
    s = yes_sub_title.lower().replace("°", " ").strip()

    # Range: "93 to 94"
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return _Strike(kind="between", low=min(lo, hi), high=max(lo, hi))

    # Single number with direction
    num_match = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not num_match:
        return None
    n = float(num_match.group(1))
    if "above" in s or "or more" in s or "greater" in s:
        return _Strike(kind="above", threshold=n)
    if "below" in s or "or less" in s or "under" in s:
        return _Strike(kind="below", threshold=n)
    return None


# ----------- Open-Meteo client -----------

_OPEN_METEO_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_FORECAST_CACHE_TTL_SEC = 600  # 10 minutes - forecasts don't change that fast
_forecast_cache: dict[str, tuple[float, list[float], list[float]]] = {}
_forecast_locks: dict[str, asyncio.Lock] = {}


async def _fetch_ensemble_temps(
    *, lat: float, lon: float, target_date: date
) -> tuple[list[float], list[float]] | None:
    """Fetch GFS-ensemble daily highs/lows for one city/date.

    Returns (highs_F, lows_F) where each list has one entry per ensemble
    member (typically 31). Cached for 10 minutes per (lat, lon, date) key.
    Returns None on transient failures.
    """
    key = f"{lat:.4f},{lon:.4f},{target_date.isoformat()}"
    now = time.time()

    # Cheap cache hit
    cached = _forecast_cache.get(key)
    if cached and (now - cached[0]) < _FORECAST_CACHE_TTL_SEC:
        return cached[1], cached[2]

    # Single-flight: avoid concurrent calls fetching the same key
    lock = _forecast_locks.setdefault(key, asyncio.Lock())
    async with lock:
        cached = _forecast_cache.get(key)
        if cached and (time.time() - cached[0]) < _FORECAST_CACHE_TTL_SEC:
            return cached[1], cached[2]

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "models": "gfs_seamless",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(_OPEN_METEO_URL, params=params)
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            log.warning("open_meteo.fetch_failed", err=str(e), lat=lat, lon=lon)
            return None

        highs: list[float] = []
        lows: list[float] = []
        daily = payload.get("daily") or {}
        for k, vals in daily.items():
            if not isinstance(vals, list) or not vals:
                continue
            v = vals[0]
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if "temperature_2m_max" in k:
                highs.append(f)
            elif "temperature_2m_min" in k:
                lows.append(f)

        if not highs and not lows:
            log.warning("open_meteo.empty_payload", lat=lat, lon=lon)
            return None

        _forecast_cache[key] = (time.time(), highs, lows)
        return highs, lows


# ----------- The model -----------


def _empirical_p_yes(members: list[float], strike: _Strike) -> float:
    """Fraction of ensemble members satisfying the strike's YES condition."""
    if not members:
        return 0.5
    if strike.kind == "above":
        n = sum(1 for m in members if m > strike.threshold)
    elif strike.kind == "below":
        n = sum(1 for m in members if m < strike.threshold)
    else:  # between
        n = sum(1 for m in members if strike.low <= m <= strike.high)
    return n / len(members)


def _ensemble_agreement(members: list[float], strike: _Strike) -> float:
    """How one-sided is the ensemble? 1.0 = unanimous, 0.5 = split 50/50."""
    if not members:
        return 0.5
    p = _empirical_p_yes(members, strike)
    return max(p, 1 - p)


@dataclass
class WeatherModel:
    enabled: bool = True

    def __post_init__(self) -> None:
        self.enabled = file_config().models.weather.enabled

    async def estimate(self, market: Market) -> ProbabilityEstimate | None:
        if not self.enabled:
            return None

        # Only handle daily temperature markets, not rain/hurricane/etc.
        event_ticker: str = market.raw.get("event_ticker", "")
        series = event_ticker.split("-", 1)[0] if event_ticker else ""
        if not series:
            return None

        # Pull city
        city = _city_from_series_ticker(series)
        if city is None:
            return None
        lat, lon, city_name = city

        # Parse strike
        strike = _parse_strike_from_yes_subtitle(market.raw.get("yes_sub_title", ""))
        if strike is None:
            return None

        # Determine target date — prefer the market's own resolution time
        target_date = self._target_date_for(market)
        if target_date is None:
            return None

        # Fetch ensemble forecast
        forecast = await _fetch_ensemble_temps(lat=lat, lon=lon, target_date=target_date)
        if forecast is None:
            return None
        highs, lows = forecast

        # Pick high-vs-low series based on the ticker
        members = highs if _is_high_market(series) else lows
        if not members:
            return None

        # Empirical probability + clip extremes (never bet 100%)
        p_yes = _empirical_p_yes(members, strike)
        p_yes = max(0.05, min(0.95, p_yes))

        agreement = _ensemble_agreement(members, strike)
        # Confidence scales with agreement but never above 0.85 — even a unanimous
        # forecast can be wrong, and Kelly sizing punishes overconfidence
        confidence = min(0.85, agreement)

        metric_label = "high" if _is_high_market(series) else "low"
        reason = (
            f"{city_name} {metric_label} on {target_date.isoformat()} "
            f"strike={strike.kind}:{strike.threshold or (strike.low, strike.high)} "
            f"members={len(members)} mean={sum(members)/len(members):.1f}F "
            f"-> p_yes={p_yes:.3f} agreement={agreement:.2f}"
        )
        return ProbabilityEstimate(
            p_yes=p_yes,
            confidence=confidence,
            reason=reason,
        )

    @staticmethod
    def _target_date_for(market: Market) -> date | None:
        """Resolve the *date* the temperature is measured on.

        Kalshi sets `expected_expiration_time` to the morning AFTER the day
        being measured (e.g. a Monday-high market expires Tue 19:00 UTC).
        For our purposes we just need the calendar date stamped in the
        ticker (`...-26MAY05-...` -> 2026-05-05) — that's the one Open-Meteo
        cares about.
        """
        event_ticker: str = market.raw.get("event_ticker", "")
        # Format: KXHIGHTDAL-26MAY05  (YY MMM DD)
        m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
        if m:
            yy, mon_abbr, dd = m.group(1), m.group(2), m.group(3)
            month_map = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,
                         "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10,
                         "NOV": 11, "DEC": 12}
            mon = month_map.get(mon_abbr)
            if mon:
                try:
                    return date(2000 + int(yy), mon, int(dd))
                except ValueError:
                    pass
        # Fallback: parse from expected_expiration_time and subtract one day
        exp = market.raw.get("expected_expiration_time")
        if exp:
            try:
                dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                return dt.date()
            except ValueError:
                pass
        return None
