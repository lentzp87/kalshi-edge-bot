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
