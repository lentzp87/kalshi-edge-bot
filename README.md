# Kalshi Edge Bot

A modular prediction-market trading bot for Kalshi. Built around the principle:
**trade mispriced probabilities, not price levels.**

## What this is

This is a **skeleton**. Every component runs end-to-end out of the box in
paper-trading mode against Kalshi's demo environment, with safe stub models
that produce no signals. Your job is to fill in the model logic and the
risk thresholds you actually want to live with.

## Architecture

```
            ┌──────────────┐
            │   Scanner    │  pulls markets, filters by price/liquidity/expiry
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │    Models    │  per-category probability estimators
            └──────┬───────┘  (weather, econ, sports, crypto, politics)
                   ▼
            ┌──────────────┐
            │   Decision   │  edge = model_prob - market_prob
            └──────┬───────┘  signal only if edge >= threshold
                   ▼
            ┌──────────────┐
            │     Risk     │  position sizing, daily loss cap, kill switch
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  Execution   │  limit orders, TP/SL/time exits
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │   Journal    │  every signal + fill + exit logged to SQLite
            └──────┬───────┘
                   ▼
            ┌──────────────┐
            │  Dashboard   │  FastAPI: /health /positions /pnl /trades
            └──────────────┘
```

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in KALSHI_API_KEY_ID and KALSHI_API_PRIVATE_KEY
python -m src.main         # runs the loop in paper mode
```

In another terminal:

```bash
python -m src.dashboard    # http://localhost:8000
```

## Going live

1. Run in paper mode for at least 2 weeks. Look at `data/trades.db`
   and confirm realized edge matches predicted edge. If it doesn't,
   your model is wrong — fix the model, do not flip to live.
2. In `config.yaml` set `mode: live`.
3. Start with `risk.max_position_size: 10` (i.e. $10 trades). Scale only
   after 100+ live trades show the same realized edge as paper.

## Deploying to Render

`render.yaml` is included. Push this repo to GitHub, then in Render:
"New > Blueprint" and point at the repo. It will create two services:

- `kalshi-bot-worker` — runs `python -m src.main`
- `kalshi-bot-dashboard` — runs `python -m src.dashboard`

Set the env vars in the Render dashboard (`KALSHI_API_KEY_ID`,
`KALSHI_API_PRIVATE_KEY`, `MODE=paper` to start). The SQLite journal
lives on a Render disk so it survives restarts.

## Honest warnings

- 5%/day on $2K compounds to ~$14M/year. That number does not exist
  in retail prediction markets. Treat this as research, not income.
- The biggest risk is not the bot — it's you raising position sizes
  after a winning streak. The risk engine enforces caps; do not
  edit them while emotional.
- Kalshi takes a fee on every trade. The realized-edge tracker in
  `journal.py` already nets fees out. Trust that number, not the
  pre-fee number.
- Sports markets are highly efficient. They're wired in but disabled
  by default in `config.yaml`. Leave them off until you've proven
  edge somewhere softer.

## Repo layout

```
.
├── README.md
├── requirements.txt
├── render.yaml
├── .env.example
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── kalshi_client.py
│   ├── scanner.py
│   ├── decision.py
│   ├── execution.py
│   ├── risk.py
│   ├── journal.py
│   ├── dashboard.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── weather.py
│       ├── econ.py
│       ├── sports.py
│       ├── crypto.py
│       └── politics.py
└── data/
    └── .gitkeep
```
