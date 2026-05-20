# Liquidity Evidence Tester

A small, research-only FastAPI app for testing whether U.S. liquidity plumbing acts as a leading signal for BTC and Nasdaq/QQQ.

**v0.1.4 hotfix:** fixes target-history coverage leakage found in the second real evidence pack. If an equity/crypto data vendor returns target prices starting later than the requested analysis window, earlier release-aligned macro rows are now left as NaN instead of being backfilled with the first available target return.

It is **not** a trading bot. It has no order routing, no brokerage actions, and no alerting layer.

## What v0.1.4 does

- Pulls official macro inputs from FRED public CSV endpoints:
  - `WALCL` = Federal Reserve total assets
  - `WTREGEN` = Treasury General Account proxy
  - `RRPONTSYD` = overnight reverse repo operations
- Optionally pulls BTC/USD daily history from CoinAPI.
- Optionally pulls QQQ daily history from Massive or Alpaca.
- Aligns H.4.1-style weekly macro information to a conservative public-release clock.
- Restricts analysed rows to the requested effective tradable window, preventing pre-start macro rows from being attached to the first available target return.
- Builds `NetLiquidity = FedAssets - TGA - ONRRP`.
- Tests liquidity impulse against forward 1, 2, 4, and 8 week returns.
- Produces a downloadable evidence pack containing CSVs, `metrics.json`, and `report.md`.
- Includes demo mode using synthetic data so the app can be validated before API credentials are added.

## Key endpoints

- `/` — operator UI
- `/health` — service and credential status
- `/api/config` — active data-source configuration
- `POST /api/run/net-liquidity` — run the evidence test
- `/api/runs/latest` — latest run summary
- `/api/evidence/latest.zip` — latest evidence pack

## Environment variables

Copy `.env.example` to `.env` locally or set these in Render:

```bash
COINAPI_KEY=
ALPACA_KEY_ID=
ALPACA_SECRET_KEY=
MASSIVE_API_KEY=
DATA_DIR=/var/data
EQUITY_SOURCE=massive
EQUITY_TARGET_SYMBOL=QQQ
COINAPI_BTC_SYMBOL_ID=COINBASE_SPOT_BTC_USD
```

`EQUITY_SOURCE` can be `massive` or `alpaca`.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000` and click **Run demo mode**.

## Render deployment

1. Upload this repo to GitHub.
2. Create a new Render Web Service from the repo.
3. Use Docker environment.
4. Add a persistent disk mounted at `/var/data`.
5. Add these secret environment variables:
   - `COINAPI_KEY`
   - `MASSIVE_API_KEY` or Alpaca keys
   - `ALPACA_KEY_ID`
   - `ALPACA_SECRET_KEY`
6. Deploy.
7. Open `/health`.
8. Open `/` and run demo mode first.
9. Run with configured APIs.
10. Download `/api/evidence/latest.zip` and upload it back for interpretation.

## Evidence discipline

Do not treat a positive backtest as permission to trade. The first promotion gate should be:

1. No look-ahead timing failure.
2. Expected sign across more than one horizon.
3. Quintile spread positive after costs.
4. Regression coefficient sign-stable.
5. Out-of-sample directional result better than chance.
6. Live-shadow evidence later agrees with the historical result.

## Known v1 limitations

- U.S. equity holiday calendars are approximated as weekdays only.
- FRED chart CSV endpoint is used for simplicity; ALFRED vintage-aware data is not yet implemented.
- CFTC and VVIX/VIX adapters are included as extension modules but the operator UI currently focuses on the first priority module: net liquidity → BTC/QQQ.
- Regression p-values use a normal approximation in the lightweight in-app estimator; this is acceptable for screening/falsification but not a substitute for a full econometrics review.
- Massive and Alpaca endpoint entitlements vary by account; check your dashboard if a request fails.


## v0.1.2 hotfix

- Fixed CoinAPI OHLCV parsing under pandas 2.x by decoding response bytes before building a DataFrame.
- Added a regression test proving CoinAPI byte payloads no longer trigger `Expected file path name or file-like object, got <class bytes> type`.

## v0.1.3 hotfix

- Fixed analysis-window leakage: FRED macro history can begin before the requested start date for rolling context, but only rows with `effective_trade_at_utc` inside the requested window are analysed.
- Adds target-specific `*_net_liquidity_features_analysis_window.csv` files.
- Renames the full macro audit file to `net_liquidity_features_full_context_unaligned.csv`.
- Adds a regression test that catches pre-start macro rows being attached to first available target returns.


## v0.1.4 hotfix

- Adds a weekly target-return merge tolerance so missing early target history cannot be backfilled across years.
- Adds `target_anchor_utc`, `matched_target_rows`, and target coverage warnings to evidence packs.
- Keeps full macro context for rolling z-score reconstruction while preventing target-history leakage.
