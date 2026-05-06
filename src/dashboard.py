"""FastAPI dashboard.

Endpoints:
  GET /health              liveness (used by Render health check)
  GET /positions           currently open
  GET /pnl                 daily P&L for last 30 days
  GET /trades?limit=100    recent trade log
  GET /edge                realized vs predicted edge buckets
  GET /stats               aggregate metrics (clean trades only)
  GET /                    full dashboard UI (auto-refreshing)

In production this is mounted by src/main.py inside the same process as
the trading loop. For standalone local runs: `python -m src.dashboard`.
"""

from __future__ import annotations

import os
from collections import defaultdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .config import env_config, file_config
from .journal import Journal

app = FastAPI(title="Kalshi Edge Bot")
_journal = Journal()


# ----- Helpers --------------------------------------------------------------

def _is_clean_trade(t: dict) -> bool:
    """Filter out trades with corrupted P&L from the early-deploy bug era.
    A long can never lose more than ~size_usd (plus a small fee buffer).
    """
    pnl = t.get("pnl_usd")
    size = t.get("size_usd") or 0
    if pnl is None:
        return False
    # Generous bound: anything beyond 3x position size is corrupt data.
    return abs(pnl) <= max(size * 3.0, 200.0)


# ----- Endpoints ------------------------------------------------------------

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


@app.get("/stats")
def stats() -> dict:
    """Aggregate metrics. Filters out the corrupt early-deploy trades."""
    raw = _journal.recent_trades(limit=10000)
    closed = [t for t in raw if t.get("closed_ts")]
    clean = [t for t in closed if _is_clean_trade(t)]

    if not clean:
        return {
            "total_pnl": 0.0, "n_trades": 0, "n_wins": 0, "n_losses": 0,
            "win_rate": 0.0, "avg_pnl": 0.0, "avg_edge_bp": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "open_positions": len(_journal.open_positions()),
            "filtered_out": len(closed) - len(clean),
            "pnl_curve": [], "by_category": {}, "edge_calibration": [],
        }

    chronological = sorted(clean, key=lambda t: t["closed_ts"])
    total = sum(t["pnl_usd"] for t in chronological)
    wins = [t for t in chronological if t["pnl_usd"] > 0]
    losses = [t for t in chronological if t["pnl_usd"] <= 0]
    n = len(chronological)
    avg_edge_pp = sum((t.get("edge") or 0) for t in chronological) / n

    # Cumulative curve over closed trades
    curve, cum = [], 0.0
    for t in chronological:
        cum += t["pnl_usd"]
        curve.append({
            "ts": t["closed_ts"],
            "cum": round(cum, 2),
            "pnl": round(t["pnl_usd"], 2),
            "ticker": t.get("ticker", ""),
        })

    # Edge calibration: bucket trades by predicted edge (5pp buckets).
    bucket_data: dict[int, dict] = defaultdict(lambda: {"n": 0, "predicted": 0.0, "realized": 0.0, "wins": 0})
    for t in chronological:
        e = (t.get("edge") or 0)
        size = t.get("size_usd") or 0
        # Predicted dollar edge = edge_pp * size_usd
        predicted_usd = e * size
        b = int(round(e * 100 / 5)) * 5  # nearest 5pp bucket
        bd = bucket_data[b]
        bd["n"] += 1
        bd["predicted"] += predicted_usd
        bd["realized"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            bd["wins"] += 1
    edge_calibration = [
        {
            "bucket_pp": k,
            "n": v["n"],
            "predicted": round(v["predicted"] / v["n"], 2),
            "realized": round(v["realized"] / v["n"], 2),
            "win_rate": round(v["wins"] / v["n"], 3),
        }
        for k, v in sorted(bucket_data.items())
    ]

    # Per-ticker-prefix (proxy for category) breakdown
    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in chronological:
        ticker = t.get("ticker", "")
        # First segment of ticker = series_ticker (e.g. KXHIGHTDAL)
        series = ticker.split("-", 1)[0] if ticker else "UNKNOWN"
        bd = by_cat[series]
        bd["n"] += 1
        bd["pnl"] += t["pnl_usd"]
        if t["pnl_usd"] > 0:
            bd["wins"] += 1
    by_category = [
        {
            "series": k, "n": v["n"], "pnl": round(v["pnl"], 2),
            "win_rate": round(v["wins"] / v["n"], 3),
        }
        for k, v in sorted(by_cat.items(), key=lambda kv: -abs(kv[1]["pnl"]))
    ]

    return {
        "total_pnl": round(total, 2),
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": round(len(wins) / n, 3),
        "avg_pnl": round(total / n, 2),
        "avg_edge_bp": round(avg_edge_pp * 100, 1),
        "best_trade": round(max(t["pnl_usd"] for t in chronological), 2),
        "worst_trade": round(min(t["pnl_usd"] for t in chronological), 2),
        "open_positions": len(_journal.open_positions()),
        "filtered_out": len(closed) - len(clean),
        "pnl_curve": curve,
        "by_category": by_category,
        "edge_calibration": edge_calibration,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    env = env_config()
    cfg = file_config()
    return _render_dashboard(
        mode=env.mode,
        kalshi_env=env.kalshi_env,
        bankroll=cfg.bankroll_usd,
        min_edge_bp=int(cfg.decision.min_edge * 100),
        min_entry_price=cfg.decision.min_entry_price,
    )


# ----- HTML/JS dashboard ----------------------------------------------------

def _render_dashboard(*, mode: str, kalshi_env: str, bankroll: float,
                     min_edge_bp: int, min_entry_price: float) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalshi Edge Bot</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root {{
  --bg: #0b0d10;
  --panel: #14171c;
  --panel-2: #1b2027;
  --border: #232931;
  --text: #e8ecef;
  --text-dim: #8a93a0;
  --accent: #4ade80;
  --danger: #f87171;
  --warn: #fbbf24;
  --info: #60a5fa;
}}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; }}
a {{ color: var(--info); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

.shell {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

.header {{ display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
.header h1 {{ font-size: 20px; margin: 0; font-weight: 600; }}
.header .subtitle {{ color: var(--text-dim); font-size: 13px; }}
.badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{ padding: 4px 10px; border-radius: 999px; font-size: 11px; font-weight: 500;
  letter-spacing: 0.4px; text-transform: uppercase; background: var(--panel-2); border: 1px solid var(--border); }}
.badge.paper {{ color: var(--info); border-color: rgba(96,165,250,0.3); }}
.badge.live {{ color: var(--danger); border-color: rgba(248,113,113,0.3); }}
.badge.prod {{ color: var(--warn); border-color: rgba(251,191,36,0.3); }}
.badge.demo {{ color: var(--text-dim); }}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; margin: 20px 0; }}
.kpi {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.kpi .label {{ color: var(--text-dim); font-size: 11px; text-transform: uppercase;
  letter-spacing: 0.6px; margin-bottom: 6px; }}
.kpi .value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.kpi .sub {{ font-size: 11px; color: var(--text-dim); margin-top: 4px; font-variant-numeric: tabular-nums; }}
.kpi.pos .value {{ color: var(--accent); }}
.kpi.neg .value {{ color: var(--danger); }}

.grid {{ display: grid; gap: 16px; margin-bottom: 16px; }}
.grid.cols-2 {{ grid-template-columns: 1fr 1fr; }}
@media (max-width: 980px) {{ .grid.cols-2 {{ grid-template-columns: 1fr; }} }}

.panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; }}
.panel .title {{ font-size: 13px; color: var(--text-dim); text-transform: uppercase;
  letter-spacing: 0.6px; margin: 0 0 12px 0; display: flex; align-items: center; justify-content: space-between; }}

.chart-wrap {{ position: relative; height: 280px; }}
.chart-wrap.tall {{ height: 320px; }}

table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ font-weight: 500; color: var(--text-dim); font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.4px; }}
tr:last-child td {{ border-bottom: none; }}
td.num {{ text-align: right; }}
td.pos {{ color: var(--accent); }}
td.neg {{ color: var(--danger); }}
td.muted {{ color: var(--text-dim); }}
td.ticker {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }}
.scroll {{ max-height: 380px; overflow-y: auto; }}

.empty {{ color: var(--text-dim); text-align: center; padding: 20px 0; font-style: italic; }}

.refresh-pill {{ font-size: 11px; color: var(--text-dim); }}
.dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: var(--accent); margin-right: 6px; vertical-align: middle;
  box-shadow: 0 0 6px var(--accent); animation: pulse 2s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} }}

.footer {{ color: var(--text-dim); font-size: 11px; padding: 18px 0;
  text-align: center; border-top: 1px solid var(--border); margin-top: 16px; }}
.footer code {{ background: var(--panel-2); padding: 2px 6px; border-radius: 4px; font-size: 11px; }}

.cfg {{ display: flex; gap: 16px; flex-wrap: wrap; font-size: 12px; color: var(--text-dim); }}
.cfg span {{ background: var(--panel-2); border: 1px solid var(--border); padding: 4px 10px;
  border-radius: 6px; }}
.cfg b {{ color: var(--text); font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<div class="shell">

  <div class="header">
    <div>
      <h1>Kalshi Edge Bot</h1>
      <div class="subtitle">
        <span class="dot"></span><span class="refresh-pill" id="refresh-status">connecting...</span>
      </div>
    </div>
    <div class="badges">
      <span class="badge {('paper' if mode == 'paper' else 'live')}">{mode}</span>
      <span class="badge {('prod' if kalshi_env == 'prod' else 'demo')}">{kalshi_env}</span>
      <span class="badge">bankroll ${bankroll:.0f}</span>
      <span class="badge">min edge {min_edge_bp}bp</span>
      <span class="badge">min entry ${min_entry_price:.2f}</span>
    </div>
  </div>

  <div class="kpis" id="kpis">
    <div class="kpi"><div class="label">Total P&amp;L</div><div class="value" id="kpi-pnl">—</div><div class="sub" id="kpi-pnl-sub"></div></div>
    <div class="kpi"><div class="label">Trades</div><div class="value" id="kpi-trades">—</div><div class="sub" id="kpi-trades-sub"></div></div>
    <div class="kpi"><div class="label">Win Rate</div><div class="value" id="kpi-winrate">—</div><div class="sub" id="kpi-winrate-sub"></div></div>
    <div class="kpi"><div class="label">Avg Edge</div><div class="value" id="kpi-edge">—</div><div class="sub">predicted</div></div>
    <div class="kpi"><div class="label">Open Positions</div><div class="value" id="kpi-open">—</div><div class="sub">capacity 15</div></div>
    <div class="kpi"><div class="label">Best / Worst</div><div class="value" id="kpi-bestworst" style="font-size:14px">—</div><div class="sub">single-trade range</div></div>
  </div>

  <div class="panel">
    <div class="title">Cumulative P&amp;L <span id="curve-meta" style="text-transform:none;letter-spacing:0;color:var(--text-dim);font-weight:400"></span></div>
    <div class="chart-wrap tall"><canvas id="chart-curve"></canvas></div>
  </div>

  <div class="grid cols-2" style="margin-top:16px">
    <div class="panel">
      <div class="title">Edge Calibration <span style="text-transform:none;letter-spacing:0;color:var(--text-dim);font-weight:400">predicted vs realized $ per trade, bucketed by edge</span></div>
      <div class="chart-wrap"><canvas id="chart-calibration"></canvas></div>
    </div>
    <div class="panel">
      <div class="title">Open Positions <span id="open-count" style="text-transform:none;letter-spacing:0;color:var(--text-dim);font-weight:400"></span></div>
      <div class="scroll">
        <table>
          <thead><tr><th>Ticker</th><th>Side</th><th class="num">Size</th><th class="num">Fill</th><th class="num">Edge</th></tr></thead>
          <tbody id="tbody-open"><tr><td colspan="5" class="empty">loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="grid cols-2" style="margin-top:16px">
    <div class="panel">
      <div class="title">Recent Trades <span id="trades-count" style="text-transform:none;letter-spacing:0;color:var(--text-dim);font-weight:400"></span></div>
      <div class="scroll">
        <table>
          <thead><tr><th>Ticker</th><th>Side</th><th class="num">Edge</th><th class="num">Fill</th><th class="num">Exit</th><th class="num">P&amp;L</th><th>Why</th></tr></thead>
          <tbody id="tbody-trades"><tr><td colspan="7" class="empty">loading...</td></tr></tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <div class="title">By Series <span style="text-transform:none;letter-spacing:0;color:var(--text-dim);font-weight:400">where we&rsquo;ve traded</span></div>
      <div class="scroll">
        <table>
          <thead><tr><th>Series</th><th class="num">Trades</th><th class="num">Win%</th><th class="num">P&amp;L</th></tr></thead>
          <tbody id="tbody-cat"><tr><td colspan="4" class="empty">loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:16px">
    <div class="title">Daily P&amp;L</div>
    <table>
      <thead><tr><th>Date</th><th class="num">Trades</th><th class="num">P&amp;L</th></tr></thead>
      <tbody id="tbody-daily"><tr><td colspan="3" class="empty">loading...</td></tr></tbody>
    </table>
  </div>

  <div class="footer">
    Auto-refreshes every 15s &middot;
    <a href="/stats">/stats</a> &middot;
    <a href="/edge">/edge</a> &middot;
    <a href="/trades">/trades</a> &middot;
    <a href="/positions">/positions</a> &middot;
    <a href="/pnl">/pnl</a>
  </div>

</div>

<script>
const fmt$ = n => (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
const fmtPct = n => (n * 100).toFixed(1) + '%';
const fmtTime = iso => {{
  if (!iso) return '';
  try {{ return new Date(iso).toLocaleString(undefined, {{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}}); }}
  catch {{ return iso; }}
}};
const cls = n => n > 0 ? 'pos' : (n < 0 ? 'neg' : 'muted');

let curveChart, calibChart;

function buildCurveChart(curve) {{
  const ctx = document.getElementById('chart-curve');
  const labels = curve.map(p => fmtTime(p.ts));
  const values = curve.map(p => p.cum);
  const tooltips = curve.map(p => p.ticker + ' ' + (p.pnl >= 0 ? '+' : '') + p.pnl.toFixed(2));
  if (curveChart) {{
    curveChart.data.labels = labels;
    curveChart.data.datasets[0].data = values;
    curveChart.data.datasets[0].tooltips = tooltips;
    curveChart.update('none');
    return;
  }}
  curveChart = new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets: [{{
      label: 'Cumulative P&L',
      data: values,
      tooltips,
      borderColor: '#4ade80',
      backgroundColor: 'rgba(74,222,128,0.08)',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.18,
      fill: true,
    }}] }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const tt = ctx.dataset.tooltips ? ctx.dataset.tooltips[ctx.dataIndex] : '';
              return [`Cum: ${{fmt$(ctx.parsed.y)}}`, tt].filter(Boolean);
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8a93a0', maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#8a93a0', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
      }}
    }}
  }});
}}

function buildCalibChart(buckets) {{
  const ctx = document.getElementById('chart-calibration');
  const labels = buckets.map(b => b.bucket_pp + 'bp');
  const predicted = buckets.map(b => b.predicted);
  const realized = buckets.map(b => b.realized);
  if (calibChart) {{
    calibChart.data.labels = labels;
    calibChart.data.datasets[0].data = predicted;
    calibChart.data.datasets[1].data = realized;
    calibChart.update('none');
    return;
  }}
  calibChart = new Chart(ctx, {{
    type: 'bar',
    data: {{ labels, datasets: [
      {{ label: 'Predicted $', data: predicted, backgroundColor: 'rgba(96,165,250,0.55)', borderRadius: 4 }},
      {{ label: 'Realized $',  data: realized,  backgroundColor: 'rgba(74,222,128,0.55)', borderRadius: 4 }},
    ] }},
    options: {{
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: {{
        legend: {{ labels: {{ color: '#e8ecef', boxWidth: 12, font: {{ size: 11 }} }} }},
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8a93a0' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#8a93a0', callback: v => '$' + v.toFixed(0) }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
      }}
    }}
  }});
}}

function renderOpen(positions) {{
  document.getElementById('open-count').textContent = positions.length + ' open';
  const tb = document.getElementById('tbody-open');
  if (!positions.length) {{ tb.innerHTML = '<tr><td colspan="5" class="empty">none</td></tr>'; return; }}
  tb.innerHTML = positions.map(p => `
    <tr>
      <td class="ticker">${{p.ticker}}</td>
      <td>${{p.side}}</td>
      <td class="num">$${{(p.size_usd||0).toFixed(0)}}</td>
      <td class="num">${{(p.fill_price||0).toFixed(2)}}</td>
      <td class="num">${{((p.edge||0)*100).toFixed(1)}}bp</td>
    </tr>`).join('');
}}

function renderTrades(rows) {{
  document.getElementById('trades-count').textContent = rows.length + ' shown';
  const tb = document.getElementById('tbody-trades');
  const closed = rows.filter(r => r.closed_ts && r.pnl_usd != null);
  if (!closed.length) {{ tb.innerHTML = '<tr><td colspan="7" class="empty">no closed trades yet</td></tr>'; return; }}
  // Hide bug-era trades with absurd P&L
  const safe = closed.filter(r => Math.abs(r.pnl_usd) <= Math.max(3 * (r.size_usd||0), 200));
  tb.innerHTML = safe.slice(0, 50).map(r => `
    <tr>
      <td class="ticker">${{r.ticker}}</td>
      <td>${{r.side}}</td>
      <td class="num">${{((r.edge||0)*100).toFixed(1)}}bp</td>
      <td class="num">${{(r.fill_price||0).toFixed(2)}}</td>
      <td class="num">${{(r.exit_price||0).toFixed(2)}}</td>
      <td class="num ${{cls(r.pnl_usd)}}">${{fmt$(r.pnl_usd)}}</td>
      <td class="muted">${{r.exit_reason||''}}</td>
    </tr>`).join('');
}}

function renderCategory(byCat) {{
  const tb = document.getElementById('tbody-cat');
  if (!byCat.length) {{ tb.innerHTML = '<tr><td colspan="4" class="empty">none</td></tr>'; return; }}
  tb.innerHTML = byCat.map(r => `
    <tr>
      <td class="ticker">${{r.series}}</td>
      <td class="num">${{r.n}}</td>
      <td class="num">${{(r.win_rate*100).toFixed(0)}}%</td>
      <td class="num ${{cls(r.pnl)}}">${{fmt$(r.pnl)}}</td>
    </tr>`).join('');
}}

function renderDaily(dailyMap) {{
  const tb = document.getElementById('tbody-daily');
  // Hide the big buggy day if it shows up: any |pnl| > $1000 is from the early bug era
  const entries = Object.entries(dailyMap || {{}})
    .filter(([d, v]) => Math.abs(v.pnl||0) <= 1000)
    .sort((a,b) => b[0].localeCompare(a[0]));
  if (!entries.length) {{ tb.innerHTML = '<tr><td colspan="3" class="empty">no closed trades yet</td></tr>'; return; }}
  tb.innerHTML = entries.map(([d, v]) => `
    <tr><td>${{d}}</td><td class="num">${{v.n}}</td><td class="num ${{cls(v.pnl)}}">${{fmt$(v.pnl)}}</td></tr>
  `).join('');
}}

function renderKpis(s) {{
  const pnl = s.total_pnl || 0;
  const e1 = document.getElementById('kpi-pnl');
  e1.textContent = fmt$(pnl);
  e1.parentElement.classList.toggle('pos', pnl > 0);
  e1.parentElement.classList.toggle('neg', pnl < 0);
  document.getElementById('kpi-pnl-sub').textContent = (s.filtered_out > 0)
    ? `excluded ${{s.filtered_out}} corrupt trades` : 'clean trades only';

  document.getElementById('kpi-trades').textContent = s.n_trades;
  document.getElementById('kpi-trades-sub').textContent = `${{s.n_wins}}W / ${{s.n_losses}}L`;

  document.getElementById('kpi-winrate').textContent = (s.win_rate*100).toFixed(0) + '%';
  document.getElementById('kpi-winrate-sub').textContent = `avg ${{fmt$(s.avg_pnl)}} per trade`;

  document.getElementById('kpi-edge').textContent = (s.avg_edge_bp||0).toFixed(0) + 'bp';

  document.getElementById('kpi-open').textContent = s.open_positions;

  const bw = `${{fmt$(s.best_trade)}} / ${{fmt$(s.worst_trade)}}`;
  document.getElementById('kpi-bestworst').textContent = bw;

  // Curve metadata
  const meta = document.getElementById('curve-meta');
  meta.textContent = s.pnl_curve.length
    ? `${{s.n_trades}} trades · range ${{fmt$(s.worst_trade)}} ... ${{fmt$(s.best_trade)}}`
    : 'no closed trades yet';
}}

async function refresh() {{
  try {{
    const [stats, openPos, tradesRows, daily] = await Promise.all([
      fetch('/stats').then(r => r.json()),
      fetch('/positions').then(r => r.json()),
      fetch('/trades?limit=200').then(r => r.json()),
      fetch('/pnl').then(r => r.json()),
    ]);
    renderKpis(stats);
    buildCurveChart(stats.pnl_curve);
    buildCalibChart(stats.edge_calibration);
    renderCategory(stats.by_category);
    renderOpen(openPos);
    renderTrades(tradesRows);
    renderDaily(daily);
    document.getElementById('refresh-status').textContent = 'updated ' + new Date().toLocaleTimeString();
  }} catch (e) {{
    document.getElementById('refresh-status').textContent = 'fetch error: ' + e.message;
  }}
}}

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""


def main() -> None:
    env = env_config()
    port = int(os.environ.get("PORT", env.dashboard_port))
    uvicorn.run("src.dashboard:app", host=env.dashboard_host, port=port, reload=False)


if __name__ == "__main__":
    main()
