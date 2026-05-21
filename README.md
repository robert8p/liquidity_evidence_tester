# Liquidity Evidence Tester

Research-only FastAPI app for testing timestamp-aligned market relationships. It is **not** a trading bot, does not place orders, and does not send alerts.

## Current modules

1. **U.S. net liquidity → BTC / QQQ**
   - Signal: Fed assets − Treasury General Account − ON RRP.
   - Uses conservative H.4.1-style release alignment.
   - v0.2.0 evidence did not validate the standalone thesis, but the module is retained for reruns and audit.

2. **CFTC JPY positioning → USD/JPY**
   - Signal: CFTC Traders in Financial Futures leveraged-fund Japanese-yen positioning.
   - Treats CFTC positions as Tuesday observations released Friday 15:30 ET.
   - Uses FRED USD/JPY daily history (`DEXJPUS`) as the initial public target series.

## Deploy on Render

Use Docker. Add a persistent disk mounted at `/var/data` and set:

```bash
DATA_DIR=/var/data
COINAPI_KEY=...
MASSIVE_API_KEY=...
# optional if using Alpaca for QQQ
EQUITY_SOURCE=alpaca
ALPACA_KEY_ID=...
ALPACA_SECRET_KEY=...
```

CFTC/FRED module does not require a paid credential.

## Operator workflow

1. Open `/health`.
2. Open `/`.
3. Run demo mode first for either module.
4. Run the configured historical test.
5. Download `/api/evidence/latest.zip`.
6. Upload the evidence ZIP back to ChatGPT for interpretation.

## Evidence gates

The app applies conservative validation labels. A candidate fails if it lacks statistical support, fails to beat always-long/always-short baselines, has materially one-sided predictions, or has poor target coverage.

Positive historical results are only candidates for deeper validation, not permission to trade.

## v0.3.1 note

The CFTC JPY module now anchors USD/JPY forward returns to the first available daily price after the conservative effective FX timestamp. v0.3.0 used weekly Friday anchoring, which was safe from look-ahead bias but skipped several tradable days after the CFTC release. This version is the correct next evidence run for judging the JPY positioning thesis.
