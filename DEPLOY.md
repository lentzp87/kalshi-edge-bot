# Deploying to Render

Follow these in order. Don't skip the local smoke test.

## 0. Prereqs

- A Kalshi account with API key + private key downloaded (`.pem` file)
- A GitHub account
- A Render account (free signup, but the bot needs the **Starter** plan
  for persistent disk — $7/mo)

## 1. Local smoke test (15 min)

You want to confirm the loop boots, hits Kalshi's demo API, and writes
a journal entry — before paying Render.

```bash
cd <this folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p secrets
cp ~/Downloads/kalshi_private_key.pem secrets/   # wherever you saved it
cp .env.example .env
# edit .env: set KALSHI_API_KEY_ID, KALSHI_API_PRIVATE_KEY_PATH, leave MODE=paper, KALSHI_ENV=demo

python -m src.main
```

You should see log lines like `bot.start mode=paper kalshi_env=demo ...`
followed by HTTP requests to `demo-api.kalshi.co`. Open
`http://localhost:8000` in a browser — empty dashboard, but it loads.

If you see `kalshi.bad_private_key` or `kalshi.no_private_key`, the auth
isn't wired right. Fix that before going further.

If you see HTTP 401s from Kalshi, the signed-header auth is being
rejected. Verify the API key id + the private key match the same row
in the Kalshi UI.

If everything looks healthy for 5 min, kill it (Ctrl+C) and continue.

## 2. Push to GitHub

```bash
git init
git add .
git commit -m "Initial kalshi edge bot skeleton"
gh repo create kalshi-edge-bot --private --source=. --push
# or: create the repo on github.com manually and `git push origin main`
```

`.gitignore` already excludes `.env`, `secrets/`, and `data/*.db` so you
won't leak credentials.

## 3. Render: blueprint deploy

1. Render dashboard → **New** → **Blueprint**
2. Connect the GitHub repo `kalshi-edge-bot`
3. Render reads `render.yaml` and proposes a service called
   `kalshi-edge-bot`. Confirm.
4. **Set the secret env vars** on the next screen (Render won't read
   them from `render.yaml` because they're marked `sync: false`):

   | Key | Value |
   |-----|-------|
   | `KALSHI_API_KEY_ID` | the id string from Kalshi |
   | `KALSHI_API_PRIVATE_KEY` | full PEM contents, newlines escaped as `\n` |

   To prep the PEM:
   ```bash
   awk '{printf "%s\\n", $0}' secrets/kalshi_private_key.pem | pbcopy
   ```
   Then paste into the Render env var field. The loader in
   `kalshi_client.py` un-escapes the `\n`s back to real newlines.

5. Click **Apply**. Render builds, installs deps, starts the bot,
   mounts the 1GB disk at `/var/data`, and exposes the dashboard at
   `https://kalshi-edge-bot.onrender.com` (or your assigned subdomain).
6. Watch the build + first 100 log lines. Confirm:
   - `bot.start mode=paper kalshi_env=demo`
   - `dashboard.start port=10000` (or whatever Render assigned)
   - `/health` returns 200 (Render's health check passes)

## 4. Verification (do this for at least 14 days)

- Visit your Render URL daily. Look at `/edge` — it shows realized P&L
  vs predicted edge per bucket. **Until those two columns track each
  other, your model is not predictive.** Don't go live.
- Check `/pnl` — paper trading should be profitable on average if your
  edge is real.
- If you're getting zero signals: that's expected with all model stubs
  returning `None`. Step 5 fixes that.

## 5. Wire up the first real model

The Weather model is the lowest-friction starting point. In
`src/models/weather.py`, replace `_forecast_distribution` with a real
call to Open-Meteo:

```python
async def _forecast_distribution(self, *, location: str, variable: str):
    # Pseudocode — adapt to the actual Kalshi market title format.
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{env_config().open_meteo_base}/forecast", params={
            "latitude": LAT, "longitude": LON,
            "hourly": "temperature_2m",
        })
        forecast = r.json()["hourly"]["temperature_2m"]
    return statistics.mean(forecast), statistics.stdev(forecast)
```

Push, Render auto-deploys, and now the weather model has an opinion.

## 6. Going live

ONLY after:
- 14+ days of paper trading
- 100+ logged trades
- Realized edge tracks predicted edge within ~30%
- You've reviewed losing trades and understand WHY they lost

Then, in the Render dashboard:
1. Set `MODE=live`
2. Set `KALSHI_ENV=prod`
3. Verify the prod base URL in `src/kalshi_client.py` `BASE_URLS["prod"]`
   matches Kalshi's current docs. They've moved endpoints before.
4. Lower `risk.max_position_size_usd` to `10` in `config.yaml` for the
   first 100 live trades.
5. Push the change. Render redeploys.
6. Watch the first hour like a hawk.

## Troubleshooting

**"Application failed to bind to PORT"** — Render's web service expects
the app to bind to `$PORT`. `src/main.py` reads it correctly; if you see
this, check the build logs to confirm `dashboard.start` ran.

**Dashboard 502 / health check failing** — the trading loop probably
crashed on boot before uvicorn started. Check logs for an exception
during `KalshiClient()` init (usually a key problem).

**Disk full** — bump `sizeGB` in `render.yaml` and redeploy. SQLite
will not be the bottleneck for years at retail volume.

**Bot keeps making the same losing trade** — your model's wrong. Don't
"fix" it by adjusting risk caps; fix the model.
