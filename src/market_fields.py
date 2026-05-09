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

# Sentence-fragment words that strongly suggest we matched mid-sentence
# (e.g. tennis title "Will X win the A vs B: ..."). If a captured group
# contains any of these as a whole word, the split is rejected.
_SENTENCE_WORDS = (
    "win", "wins", "won", "beat", "beats", "the", "will", "match",
    "round", "professional", "game",
)
_SENTENCE_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_SENTENCE_WORDS) + r")\b", re.IGNORECASE,
)

# Splitters in priority order. Most Kalshi titles use ' vs ' (regular
# season), 'Game N: X at Y' (playoffs), or sometimes ' v ' or ' v. '.
_TITLE_SPLITTERS: tuple[str, ...] = (
    r"\s+vs\.?\s+",
    r"\s+at\s+",
    r"\s+v\.?\s+",
)


def _looks_like_team_name(s: str) -> bool:
    """Heuristic: a real team / player name shouldn't contain sentence
    words like 'win', 'the', 'match', and shouldn't be more than 4 words.
    """
    if not s:
        return False
    if _SENTENCE_WORD_RE.search(s):
        return False
    if len(s.split()) > 4:
        return False
    return True


def teams_from_title(title: str) -> tuple[str, str] | None:
    """Parse a Kalshi market title into (team_a, team_b).

    Handles the three observed formats:
      * 'Pittsburgh vs San Francisco Winner?'    (regular-season MLB)
      * 'Game 2: Cleveland at Detroit Winner?'   (NBA/NHL playoffs)
      * 'Carolina at Philadelphia Winner?'        (some sports use 'at')

    Sentence-style titles (tennis: 'Will X win the A vs B...') are
    rejected here — the tennis model uses teams_from_tennis_title for
    those. For team sports the convention is (away, home).
    """
    if not title:
        return None
    t = title.strip()
    # Strip trailing '?'
    t = re.sub(r"\s*\?+\s*$", "", t)
    # Strip trailing ' Winner'
    t = re.sub(r"\s+Winner\s*$", "", t, flags=re.IGNORECASE)
    # Strip 'Game N: ' playoff prefix
    t = re.sub(r"^Game\s+\d+\s*:\s*", "", t, flags=re.IGNORECASE)
    if not t:
        return None

    for pat in _TITLE_SPLITTERS:
        parts = re.split(pat, t, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not a or not b:
            continue
        # Reject obvious sentence captures (tennis-style titles)
        if not _looks_like_team_name(a) or not _looks_like_team_name(b):
            continue
        return a, b
    return None


# Sentence-shaped titles (tennis, UFC, soccer phrasings):
#   'Will Taylor Townsend win the Bouzkova vs Townsend: Round Of 64 match?'
#   'Will Carlos Alcaraz win the Alcaraz/Sinner match?'
#   'Will Jeremy Stephens win the Green vs Stephens professional MMA fight ...'
#   'Will San Diego win the St. Louis vs San Diego game ...'
#
# We extract the bit after 'win the' and split on ' vs ' / '/'. Trailing
# context word can be: match, round, fight, game, bout, tournament, ...
# or punctuation, or end-of-string.
_TENNIS_TITLE_RE = re.compile(
    r"""win\s+the\s+
        (?P<a>[\w'.\-]+(?:\s+[\w'.\-]+){0,3}?)
        \s*(?:vs\.?|/)\s*
        (?P<b>[\w'.\-]+(?:\s+[\w'.\-]+){0,3}?)
        \s*
        (?=
            [:,.]
            | \s+(?:match|round|fight|game|bout|tournament|professional|MMA|scheduled)
            | \s*$
        )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def teams_from_tennis_title(title: str) -> tuple[str, str] | None:
    """Sentence-shaped title parser. Used for tennis, UFC, and any sport
    where Kalshi phrases markets as 'Will X win the A vs B ... match/fight'.

    Returns (side_a_surname, side_b_surname).
    """
    if not title:
        return None
    m = _TENNIS_TITLE_RE.search(title)
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


# Aliases for team / player nicknames that don't share letters with
# their canonical name. These come up when Kalshi uses a colloquial
# display name (e.g. "A's") but ESPN / The Odds API use the official
# city + nickname ("Oakland Athletics"). Add entries here when a
# sports.skip.our_side_unmatched / no_sportsbook_odds log reveals one.
# Keys / values are lowercase, alphanumeric only. Direction-symmetric.
_NICKNAME_ALIASES: dict[str, set[str]] = {
    # MLB nicknames
    "as": {"athletics", "oakland", "ath"},      # Kalshi "A's" ↔ Athletics
    "athletics": {"as", "ath"},
    "ath": {"as", "athletics", "oakland"},
    # NBA acronyms (yes_sub_title prefix doesn't match city)
    "okc": {"oklahoma"},
    "lal": {"lakers"},                           # but LA Lakers usually OK via "Los"
    "nyk": {"knicks", "york"},
    "gsw": {"warriors", "golden", "state"},
    "nop": {"pelicans"},
    "sas": {"spurs"},
    # NHL acronyms
    "tbl": {"lightning", "tampa"},
    "nyr": {"rangers", "york"},
    "nyi": {"islanders"},
    "njd": {"devils", "jersey"},
    "vgk": {"vegas", "knights"},
    "lak": {"kings"},
    "sjs": {"sharks"},
    # NFL acronyms (when these come up)
    "lar": {"rams"},
    "lac": {"chargers"},
}


def _alias_match(a: str, b: str) -> bool:
    """True if a and b are linked via the nickname-alias graph."""
    return b in _NICKNAME_ALIASES.get(a, set()) or a in _NICKNAME_ALIASES.get(b, set())


def name_match(target: str, candidate: str) -> bool:
    """Loose name-equality used for matching team / player names across
    Kalshi titles, ESPN responses, and The Odds API.

    Match strategy (in order):
      1. Whole-string substring (after stripping punctuation)
         — handles "Pittsburgh" vs "Pittsburgh Pirates".
      2. Tokenized prefix match
         — handles "PHI Flyers" vs "Philadelphia" via PHI→PHIladelphia.
      3. Nickname alias lookup
         — handles "A's" vs "Oakland Athletics" via the alias table.

    Both args are normalized: punctuation stripped, uppercased, and
    split on whitespace. Substring matches require >=3 chars on the
    short side to avoid 1-2 char false positives.
    """
    if not target or not candidate:
        return False
    # Apostrophes contract (A's -> As, not A s) — keep neighboring chars
    # together. Other punctuation becomes whitespace.
    def _norm(s: str) -> str:
        s = re.sub(r"[‘’']", "", s)
        s = re.sub(r"[^\w\s]", " ", s).upper().strip()
        return s
    t = _norm(target)
    c = _norm(candidate)
    if not t or not c:
        return False
    # 1) Whole-string substring (collapsed whitespace)
    t_join = re.sub(r"\s+", " ", t)
    c_join = re.sub(r"\s+", " ", c)
    if t_join == c_join:
        return True
    if t_join in c_join or c_join in t_join:
        return True

    # 2) Tokenized: any token from one side prefix-matches a token from
    #    the other. Tokens shorter than 2 chars are dropped to avoid
    #    pathological matches; prefix match requires >= 3 chars.
    t_tokens = [tok for tok in t.split() if len(tok) >= 2]
    c_tokens = [tok for tok in c.split() if len(tok) >= 2]
    for tt in t_tokens:
        for ct in c_tokens:
            if tt == ct:
                return True
            short, long_ = (tt, ct) if len(tt) <= len(ct) else (ct, tt)
            if len(short) >= 3 and long_.startswith(short):
                return True
            # 3) Alias check at the token level
            if _alias_match(tt.lower(), ct.lower()):
                return True
    return False


def series_prefix(ticker: str) -> str:
    """Return the leading series-ticker segment (everything before the
    first '-'). E.g. 'KXMLBGAME-26MAY092105PITSF-SF' -> 'KXMLBGAME'.

    This is the only piece of the ticker we still rely on — the series
    prefix is well-defined and stable, unlike the event-code segment.
    """
    return (ticker or "").split("-", 1)[0]
