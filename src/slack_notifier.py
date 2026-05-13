"""Slack trade notifications via the existing PatBot Ops Agent app.

Two transports, picked in order:
  1. SLACK_WEBHOOK_URL  — Incoming Webhook (simplest, no token mgmt)
  2. SLACK_BOT_TOKEN + SLACK_CHANNEL — chat.postMessage API

If neither env var is set we log a single warning at startup and
no-op every send. Slack errors never crash the trading loop — every
send is wrapped, logs a warning, and returns.

Why we duck-type the message: the journal stores a `reason` string
that already encodes provider, p_yes, edge, whale class, etc. The
notifier extracts the most useful bits for a human glancing at their
phone. The full reason is appended at the bottom of the message for
completeness.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone

import httpx
import structlog

log = structlog.get_logger(__name__)


_RE_PROVIDER = re.compile(r"book\[([^\]]+)\]")
# Terminate the our= label on either " p_yes=" (sports.py / tennis.py / etc.
# all append p_yes after the side label) or " |" (next reason segment).
# Falls back gracefully if neither is present.
_RE_OUR = re.compile(r"our=(.+?)(?=\s+p_yes=|\s*\|)")
_RE_WHALE_CLASS = re.compile(r"WHALE_ALIGNED class=(\w+)")


def _env() -> tuple[str | None, str | None, str | None]:
    return (
        os.environ.get("SLACK_WEBHOOK_URL"),
        os.environ.get("SLACK_BOT_TOKEN"),
        os.environ.get("SLACK_CHANNEL"),
    )


_warned_no_config = False


def _warn_once() -> None:
    global _warned_no_config
    if not _warned_no_config:
        log.warning(
            "slack.no_config",
            msg=(
                "Neither SLACK_WEBHOOK_URL nor (SLACK_BOT_TOKEN + SLACK_CHANNEL) "
                "is set; trade notifications disabled"
            ),
        )
        _warned_no_config = True


async def _post(text: str) -> None:
    """Send `text` to Slack. No-op on missing config. Fail-silent on errors."""
    webhook, token, channel = _env()
    if not webhook and not (token and channel):
        _warn_once()
        return

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if webhook:
                r = await client.post(webhook, json={"text": text})
                if r.status_code >= 400:
                    log.warning("slack.webhook_error",
                                status=r.status_code, body=r.text[:200])
            else:
                # chat.postMessage. Channel can be a #name or ID.
                r = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json; charset=utf-8",
                    },
                    json={"channel": channel, "text": text},
                )
                data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                if not data.get("ok"):
                    log.warning("slack.api_error",
                                status=r.status_code,
                                error=data.get("error"),
                                channel=channel)
    except Exception as e:  # noqa: BLE001 - Slack hiccup must not kill bot
        log.warning("slack.send_failed", err=str(e)[:200])


# ----------------------------------------------------------------- public API

def _our_label(reason: str) -> str:
    if m := _RE_OUR.search(reason or ""):
        return m.group(1).strip()
    return ""


def _provider(reason: str) -> str:
    if m := _RE_PROVIDER.search(reason or ""):
        return m.group(1).strip()
    return ""


def _whale_tag(reason: str) -> str:
    if m := _RE_WHALE_CLASS.search(reason or ""):
        return f" 🐋 {m.group(1)}"
    return ""


def _sport_from_ticker(ticker: str) -> str:
    if not ticker:
        return "?"
    return ticker.split("-", 1)[0]


def _fire(coro) -> None:
    """Schedule a Slack send without awaiting it. The trading loop doesn't
    need to block on Slack latency. Uses the running event loop; if there
    isn't one (sync call from a non-async path), runs it on a temporary one.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # No running loop — fall back to a blocking send. Only happens
        # outside the trading loop (tests, manual scripts).
        asyncio.run(coro)


def notify_open(*, ticker: str, side: str, size_usd: float, fill_price: float,
                edge: float, reason: str) -> None:
    """Fire a 'trade opened' Slack message."""
    sport = _sport_from_ticker(ticker)
    label = _our_label(reason) or ticker
    provider = _provider(reason) or "?"
    edge_bp = edge * 100
    whale = _whale_tag(reason)
    text = (
        f"🟢 *OPENED* `{label}` ({sport}) "
        f"{side.upper()} ${size_usd:.0f} @ {fill_price:.2f}{whale}\n"
        f"edge {edge_bp:+.1f}bp · {provider}\n"
        f"`{ticker}`"
    )
    _fire(_post(text))


def notify_close(*, ticker: str, side: str, fill_price: float,
                 exit_price: float, pnl_usd: float, fees_usd: float,
                 reason: str, exit_reason: str,
                 opened_at: datetime | None = None) -> None:
    """Fire a 'trade closed' Slack message."""
    sport = _sport_from_ticker(ticker)
    label = _our_label(reason) or ticker
    emoji = "🟩" if pnl_usd > 0 else ("🟥" if pnl_usd < 0 else "⬜️")
    pnl_str = f"{pnl_usd:+.2f}"
    move_str = f"{fill_price:.2f}→{exit_price:.2f}"
    held = ""
    if opened_at:
        mins = max(0, int((datetime.now(timezone.utc) - opened_at).total_seconds() / 60))
        if mins < 60:
            held = f" · held {mins}m"
        else:
            held = f" · held {mins // 60}h {mins % 60}m"
    text = (
        f"{emoji} *CLOSED* `{label}` ({sport}) "
        f"{side.upper()} {move_str} · *${pnl_str}* "
        f"(fees ${fees_usd:.2f}){held}\n"
        f"exit: {exit_reason}\n"
        f"`{ticker}`"
    )
    _fire(_post(text))


def notify_startup(mode: str, bankroll: float) -> None:
    """One-shot startup ping so you know the bot redeployed."""
    text = f"🤖 Bot online · {mode} mode · bankroll ${bankroll:.0f}"
    _fire(_post(text))


# --------------------------------------------------------- 6h P&L digest

def _fmt_money(x: float) -> str:
    return f"${x:+.2f}" if x else "$0.00"


def _bet_label(reason: str, ticker: str) -> str:
    """Short label for a row in the digest. Falls back to the ticker tail
    if the reason doesn't carry our= info (e.g. early-version rows)."""
    label = _our_label(reason)
    if label:
        return label
    return (ticker or "").rsplit("-", 1)[-1] or "?"


def _unrealized_pnl(pos: dict) -> float | None:
    """Mark-to-market for an open position. Returns None if we don't have
    a current_mid yet (watcher hasn't sampled this position).

    Math matches Executor._exit but estimated, not realized:
      pnl = (current_mid - fill_price) * contracts  -  estimated exit fee
    Entry fee is sunk — we don't double-count it (the reported "open"
    P&L already implicitly includes the entry-fee drag because fill_price
    is what we paid).
    """
    mid = pos.get("current_mid")
    fill = pos.get("fill_price")
    contracts = pos.get("contracts")
    if mid is None or fill is None or not contracts:
        return None
    try:
        from .fee_model import fee_per_contract_dollars
        exit_fee = fee_per_contract_dollars(float(mid)) * int(contracts)
    except Exception:
        exit_fee = 0.0
    return (float(mid) - float(fill)) * int(contracts) - exit_fee


def _window_stats(closed_rows: list[dict], cutoff_iso: str) -> dict:
    """Realized stats for rows whose closed_ts >= cutoff_iso."""
    in_win = [r for r in closed_rows if r.get("closed_ts", "") >= cutoff_iso]
    n = len(in_win)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "realized": 0.0}
    realized = sum(float(r.get("pnl_usd") or 0) for r in in_win)
    wins = sum(1 for r in in_win if (r.get("pnl_usd") or 0) > 0)
    losses = sum(1 for r in in_win if (r.get("pnl_usd") or 0) < 0)
    return {"n": n, "wins": wins, "losses": losses, "realized": realized}


def build_pnl_digest(journal, *, windows: list[int] | None = None,
                     window_hours: int | None = None) -> str:
    """Multi-window P&L digest as a single Slack message.

    Default windows = [3, 6, 12, 24]. The unrealized line and top/bottom
    trades are drawn once from the widest window (so we don't repeat
    them four times). The four windows share a monospace table.

    Backwards-compat: `window_hours=N` is still accepted and produces a
    single-window digest, matching the prior /pnl_digest behavior.
    """
    from datetime import datetime, timezone, timedelta

    if windows is None and window_hours is None:
        windows = [3, 6, 12, 24]
    elif windows is None:
        windows = [int(window_hours)]  # legacy single-window call

    windows = sorted(set(int(w) for w in windows))
    max_win = max(windows)

    now = datetime.now(timezone.utc)
    cutoffs = {w: (now - timedelta(hours=w)).isoformat() for w in windows}
    max_cutoff_iso = cutoffs[max_win]

    # Pull a generous slice; we'll filter per-window below.
    rows = journal.recent_trades(limit=1000)
    relevant_closed = [
        r for r in rows
        if r.get("closed_ts") and r["closed_ts"] >= max_cutoff_iso
        and r.get("pnl_usd") is not None
    ]

    # Per-window stats
    per_window = [
        (w, _window_stats(relevant_closed, cutoffs[w])) for w in windows
    ]

    # Open positions — single snapshot, window-independent.
    open_positions = journal.open_positions()
    open_with_mtm = []
    for p in open_positions:
        u = _unrealized_pnl(p)
        if u is not None:
            open_with_mtm.append((u, p))
    open_with_mtm.sort(key=lambda x: x[0], reverse=True)
    unrealized = sum(u for u, _ in open_with_mtm)
    n_open = len(open_positions)
    n_with_mtm = len(open_with_mtm)

    # ---- Compose message ----
    lines = ["📊 *P&L snapshot*", ""]

    # Open / unrealized summary (single line)
    if n_open == 0:
        lines.append("*Open:* no positions")
    elif n_with_mtm == 0:
        lines.append(f"*Open:* {n_open} position(s) · awaiting first mid sample")
    else:
        lines.append(
            f"*Open:* {_fmt_money(unrealized)} unrealized "
            f"({n_with_mtm}/{n_open} marked)"
        )
        best_u, best_p = open_with_mtm[0]
        worst_u, worst_p = open_with_mtm[-1]
        if best_u > 0:
            lines.append(
                f"  ↗ {_bet_label(best_p.get('reason',''), best_p.get('ticker',''))} "
                f"{_fmt_money(best_u)}"
            )
        if worst_u < 0 and worst_p is not best_p:
            lines.append(
                f"  ↘ {_bet_label(worst_p.get('reason',''), worst_p.get('ticker',''))} "
                f"{_fmt_money(worst_u)}"
            )

    lines.append("")

    # Monospace window table — Slack renders ``` blocks in fixed-width
    # so columns align. Width budget: label 12, each window 11.
    def _col(s: str, w: int = 11) -> str:
        return s.rjust(w)

    header = "Window      " + "".join(_col(f"{w}h") for w in windows)
    n_row  = "Closed N    " + "".join(_col(str(s["n"])) for _w, s in per_window)
    wl_row = "W / L       " + "".join(_col(f"{s['wins']}/{s['losses']}") for _w, s in per_window)
    rp_row = "Realized    " + "".join(_col(_fmt_money(s["realized"])) for _w, s in per_window)
    net_row = "Net         " + "".join(
        _col(_fmt_money(s["realized"] + unrealized)) for _w, s in per_window
    )

    lines.append("```")
    lines.append(header)
    lines.append(n_row)
    lines.append(wl_row)
    lines.append(rp_row)
    lines.append(net_row)
    lines.append("```")

    # Top / bottom over the widest window
    by_pnl = sorted(relevant_closed, key=lambda r: float(r.get("pnl_usd") or 0), reverse=True)
    top = by_pnl[:2]
    bottom = list(reversed(by_pnl[-2:])) if len(by_pnl) >= 2 else []
    tops_text = ", ".join(
        f"{_bet_label(r.get('reason',''), r.get('ticker',''))} "
        f"{_fmt_money(float(r.get('pnl_usd') or 0))}"
        for r in top if (r.get('pnl_usd') or 0) > 0
    )
    bots_text = ", ".join(
        f"{_bet_label(r.get('reason',''), r.get('ticker',''))} "
        f"{_fmt_money(float(r.get('pnl_usd') or 0))}"
        for r in bottom if (r.get('pnl_usd') or 0) < 0
    )
    if tops_text:
        lines.append(f"🏆 *{max_win}h winners:* {tops_text}")
    if bots_text:
        lines.append(f"💀 *{max_win}h losers:* {bots_text}")

    return "\n".join(lines)


def notify_pnl_digest(journal, *, windows: list[int] | None = None,
                      window_hours: int | None = None) -> None:
    """Fire the digest to Slack. Fail-silent."""
    try:
        text = build_pnl_digest(journal, windows=windows, window_hours=window_hours)
    except Exception as e:  # noqa: BLE001
        log.warning("slack.digest_build_failed", err=str(e)[:200])
        return
    _fire(_post(text))
