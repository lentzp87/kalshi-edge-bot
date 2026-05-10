"""Detect price spreads between Kalshi and Polymarket on the same event.

For every Kalshi market we find the best-matching Polymarket market by
title-token Jaccard similarity. If the score clears `min_match_score`
and the |price spread| clears `min_abs_spread_pp`, we emit a
`CrossExchangeSpread`. Read-only — no orders are placed here. Wiring
into main.py / the dashboard is intentionally left to a follow-up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from .polymarket_client import parse_yes_price

log = structlog.get_logger(__name__)


# Common English noise that hurts more than helps for short prediction-market
# titles. We keep nouns / proper nouns / numbers / team names.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an",
    "of", "to", "in", "on", "at", "for", "by", "with",
    "and", "or",
    "vs", "v",
    "will", "win", "wins", "winner", "won",
    "be", "is", "are", "do", "does",
    "who", "what", "which", "when",
    "?", "—", "-", "–",
})

# Strip every char that isn't a word char or whitespace. This catches "?" / em
# dashes / commas / apostrophes etc. before tokenization.
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


@dataclass
class CrossExchangeSpread:
    kalshi_ticker: str
    kalshi_title: str
    kalshi_yes_price: float
    kalshi_volume: int
    polymarket_slug: str
    polymarket_question: str
    polymarket_yes_price: float
    polymarket_volume: float
    spread_pp: float            # signed: positive = Kalshi higher
    abs_spread_pp: float
    match_score: float          # 0-1, how confident the title match is
    arb_direction: str          # informational; see _arb_direction()


def _normalize_title(title: str) -> set[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stop words.
    Returns a set of meaningful tokens for Jaccard scoring.
    """
    if not title:
        return set()
    cleaned = _PUNCT_RE.sub(" ", title.lower())
    tokens = cleaned.split()
    return {t for t in tokens if t and t not in _STOP_WORDS}


def _match_score(kalshi_title: str, poly_question: str) -> float:
    """Token Jaccard score between two titles, in [0, 1].

    Jaccard = |A ∩ B| / |A ∪ B|. Cheap, symmetric, and works well for
    short titles where word ORDER matters less than word OVERLAP.
    Cosine over TF-IDF would be overkill here — most titles are 5-10
    tokens with no repetition.
    """
    a = _normalize_title(kalshi_title)
    b = _normalize_title(poly_question)
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _arb_direction(spread_pp: float) -> str:
    """Informational label for which side is cheap on which venue.

    spread_pp = kalshi_yes - polymarket_yes
      >0  Kalshi YES is more expensive; cheap side is Polymarket YES
          (or, equivalently, Kalshi NO).
      <0  Polymarket YES is more expensive; cheap side is Kalshi YES
          (or Polymarket NO).
    """
    if spread_pp > 0:
        return "buy_polymarket_yes_or_kalshi_no"
    if spread_pp < 0:
        return "buy_kalshi_yes_or_polymarket_no"
    return "no_spread"


def _poly_volume(market: dict) -> float:
    """Polymarket exposes `volume` as a string sometimes — coerce safely."""
    v = market.get("volume")
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def find_spreads(
    kalshi_markets: list,            # list of Market dataclass instances
    polymarket_markets: list[dict],
    *,
    min_match_score: float = 0.6,
    min_abs_spread_pp: float = 0.03,
    top_n: int = 50,
) -> list[CrossExchangeSpread]:
    """For each Kalshi market, find the best Polymarket match by title
    Jaccard score. If score >= min_match_score and price spread >=
    min_abs_spread_pp, emit a CrossExchangeSpread. Returns top_n by
    descending |spread|.
    """
    if not kalshi_markets or not polymarket_markets:
        return []

    # Pre-tokenize Polymarket titles once so the inner loop is cheap.
    poly_index: list[tuple[set[str], dict, float | None]] = []
    for pm in polymarket_markets:
        question = pm.get("question") or ""
        tokens = _normalize_title(question)
        if not tokens:
            continue
        yes = parse_yes_price(pm)
        if yes is None:
            continue
        poly_index.append((tokens, pm, yes))

    if not poly_index:
        return []

    spreads: list[CrossExchangeSpread] = []

    for km in kalshi_markets:
        k_title = getattr(km, "title", "") or ""
        k_tokens = _normalize_title(k_title)
        if not k_tokens:
            continue

        # Inline Jaccard against the pre-tokenized Polymarket index.
        best_score = 0.0
        best_pm: dict | None = None
        best_pm_yes: float | None = None
        for p_tokens, pm, pm_yes in poly_index:
            inter = len(k_tokens & p_tokens)
            if inter == 0:
                continue
            union = len(k_tokens | p_tokens)
            if union == 0:
                continue
            score = inter / union
            if score > best_score:
                best_score = score
                best_pm = pm
                best_pm_yes = pm_yes

        if best_pm is None or best_pm_yes is None:
            continue
        if best_score < min_match_score:
            continue

        # Use the Kalshi mid (effective bid/ask aware) as the YES price.
        # Falls back to last_price when the book is one-sided.
        try:
            k_yes = float(km.mid)
        except Exception:
            k_yes = float(getattr(km, "last_price", 0.0) or 0.0)

        spread_pp = k_yes - best_pm_yes
        abs_spread = abs(spread_pp)
        if abs_spread < min_abs_spread_pp:
            continue

        spreads.append(CrossExchangeSpread(
            kalshi_ticker=getattr(km, "ticker", ""),
            kalshi_title=k_title,
            kalshi_yes_price=k_yes,
            kalshi_volume=int(getattr(km, "volume", 0) or 0),
            polymarket_slug=best_pm.get("slug", "") or "",
            polymarket_question=best_pm.get("question", "") or "",
            polymarket_yes_price=best_pm_yes,
            polymarket_volume=_poly_volume(best_pm),
            spread_pp=spread_pp,
            abs_spread_pp=abs_spread,
            match_score=best_score,
            arb_direction=_arb_direction(spread_pp),
        ))

    spreads.sort(key=lambda s: s.abs_spread_pp, reverse=True)
    return spreads[:top_n]
