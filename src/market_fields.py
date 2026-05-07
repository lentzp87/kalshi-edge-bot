"""Helpers to read structured Kalshi market fields.

Kalshi returns rich human-readable fields on every market that obsolete
fragile ticker decoding:

  * title              — "Pittsburgh vs San Francisco Winner?"
  * yes_sub_title      — "San Francisco" (which side resolves YES)
  * no_sub_title       — opposing side
  * occurrence_datetime — ISO8601 UTC scheduled start time
  * rules_primary      — plain-English resolution rule

This module centralizes reading these fields with sensible fallbacks.
Models should call these helpers instead of regex-parsing tickers.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# "Pittsburgh vs San Francisco Winner?" -> ("Pittsburgh", "San Francisco")
# Tolerates: 'vs' or 'vs.', optional trailing 'Winner', optional '?'.
# For team sports the convention is "Away vs Home"; for tennis it's just
# the two players in seeding/alphabetical order.
_TITLE_VS_RE = re.compile(
    r"""^\s*
        (?P<a>.+?)
        \s+vs\.?\s+
        (?P<b>.+?)
        (?:\s+Winner)?
        \s*\??\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def teams_from_title(title: str) -> tuple[str, str] | None:
    """Parse 'X vs Y Winner?' into (X, Y).

    For team sports the order is (away, home). For tennis it's just the
    two players — order isn't meaningful for matching since we use
    yes_sub_title to know which side is YES.
    """
    if not title:
        return None
    m = _TITLE_VS_RE.match(title)
    if not m:
        return None
    a = m.group("a").strip()
    b = m.group("b").strip()
    if not a or not b:
        return None
    return a, b


# Fallback: parse from rules text. Looks for "X vs Y professional ... game"
# or "X vs Y match" patterns. Less reliable than title but works when
# Kalshi changes title formatting.
_RULES_VS_RE = re.compile(
    r"""\b
        (?P<a>[A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*)*)
        \s+vs\.?\s+
        (?P<b>[A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*)*)
        \s+(?:professional\s+)?(?:\w+\s+)?(?:game|match)
    """,
    re.VERBOSE,
)


def teams_from_rules(rules: str) -> tuple[str, str] | None:
    """Fallback team extraction from rules_primary."""
    if not rules:
        return None
    m = _RULES_VS_RE.search(rules)
    if not m:
        return None
    return m.group("a").strip(), m.group("b").strip()


def event_start_utc(market_raw: dict) -> datetime | None:
    """Parse occurrence_datetime (ISO8601) into a UTC-aware datetime.

    Falls back to expected_expiration_time, then close_time. Returns None
    if no usable timestamp is present.
    """
    candidates = (
        market_raw.get("occurrence_datetime"),
        market_raw.get("expected_expiration_time"),
        market_raw.get("close_time"),
    )
    for iso in candidates:
        if not iso:
            continue
        try:
            s = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, AttributeError):
            continue
    return None


# Falls back to parsing "If X wins ..." from rules_primary
_RULES_IF_WINS_RE = re.compile(
    r"^\s*If\s+(?P<who>.+?)\s+(?:wins?|beats?)\b",
    re.IGNORECASE,
)


def yes_side_name(market_raw: dict) -> str:
    """Return the human name of the side that resolves YES.

    Primary: yes_sub_title (always present in modern Kalshi markets).
    Fallback: parse rules_primary "If X wins ..." if yes_sub_title is empty.
    Returns empty string if both fail.
    """
    name = (market_raw.get("yes_sub_title") or "").strip()
    if name:
        return name
    rules = market_raw.get("rules_primary") or ""
    m = _RULES_IF_WINS_RE.match(rules)
    return m.group("who").strip() if m else ""


def name_match(target: str, candidate: str) -> bool:
    """Loose name-equality used for matching team / player names across
    Kalshi titles and book / ESPN responses.

    Match is symmetric and case-insensitive: returns True if either name
    contains the other after normalizing whitespace and punctuation.
    """
    if not target or not candidate:
        return False
    t = re.sub(r"[^\w\s]", "", target).upper().strip()
    c = re.sub(r"[^\w\s]", "", candidate).upper().strip()
    if not t or not c:
        return False
    if t == c:
        return True
    return t in c or c in t


def series_prefix(ticker: str) -> str:
    """Return the leading series-ticker segment (everything before the
    first '-'). E.g. 'KXMLBGAME-26MAY092105PITSF-SF' -> 'KXMLBGAME'.

    This is the only piece of the ticker we still rely on — the series
    prefix is well-defined and stable, unlike the event-code segment.
    """
    return (ticker or "").split("-", 1)[0]
