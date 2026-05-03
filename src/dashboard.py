"""FastAPI dashboard.

Endpoints:
  GET /health              liveness (used by Render health check)
  GET /positions           currently open
  GET /pnl                 daily P&L for last 30 days
  GET /trades?limit=100    recent trade log
  GET /edge                realized vs predicted edge buckets
  GET /                    minimal HTML overview

In production this is mounted by src/main.py inside the same process as
the trading loop. For standalone local runs: `python -m src.dashboard`.
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import env_config
from .journal import Journal

app = FastAPI(title="Kalshi Edge Bot")
_journal = Journal()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/positions")
def positions() -> list[dict]:
    return _journal.open_positions()


@app.get("/trades")
def trades(limit: int = 100) -> list[dict]:
    return _journal.recent_trades(limit=limit)


@app.get("/pnl")
def pnl() -> dict:
    return _journal.daily_pnl()


@app.get("/edge")
def edge() -> dict:
    return _journal.realized_edge_summary()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    pnl_data = _journal.daily_pnl()
    open_pos = _journal.open_positions()
    rows = "".join(
        f"<tr><td>{d}</td><td>{v['n']}</td><td>${v['pnl']:.2f}</td></tr>"
        for d, v in pnl_data.items()
    )
    open_rows = "".join(
        f"<tr><td>{p['ticker']}</td><td>{p['side']}</td><td>${p['size_usd']:.0f}</td>"
        f"<td>{p['fill_price']:.2f}</td><td>{(p['edge'] or 0) * 100:.1f}bp</td></tr>"
        for p in open_pos
    )
    return f"""
    <html><head><title>Kalshi Edge Bot</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; padding: 2rem; max-width: 900px; margin: auto; }}
      table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
      th, td {{ text-align: left; padding: 0.4rem 0.8rem; border-bottom: 1px solid #eee; }}
      th {{ background: #f7f7f7; }}
      h1, h2 {{ font-weight: 600; }}
    </style></head>
    <body>
      <h1>Kalshi Edge Bot</h1>
      <h2>Open positions ({len(open_pos)})</h2>
      <table><thead><tr><th>Ticker</th><th>Side</th><th>Size</th><th>Fill</th><th>Edge</th></tr></thead>
      <tbody>{open_rows or '<tr><td colspan=5>none</td></tr>'}</tbody></table>

      <h2>Daily P&amp;L</h2>
      <table><thead><tr><th>Date</th><th>Trades</th><th>P&amp;L</th></tr></thead>
      <tbody>{rows or '<tr><td colspan=3>no closed trades yet</td></tr>'}</tbody></table>

      <p>See also <a href="/edge">/edge</a> · <a href="/trades">/trades</a> · <a href="/positions">/positions</a></p>
    </body></html>
    """


def main() -> None:
    env = env_config()
    port = int(os.environ.get("PORT", env.dashboard_port))
    uvicorn.run("src.dashboard:app", host=env.dashboard_host, port=port, reload=False)


if __name__ == "__main__":
    main()
