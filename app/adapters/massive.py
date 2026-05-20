from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

BASE_URL = 'https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}'


def fetch_daily_bars(ticker: str, api_key: str, start: str, end: str, raw_dir: Path | None = None) -> pd.DataFrame:
    url = BASE_URL.format(ticker=ticker, start=start, end=end)
    params = {'adjusted': 'true', 'sort': 'asc', 'limit': '50000', 'apiKey': api_key}
    content = get_bytes(url, params=params)
    if raw_dir:
        snapshot_payload(raw_dir, f'massive_{ticker}', url, content, suffix='json')
    payload = json.loads(content.decode('utf-8'))
    rows = payload.get('results', []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['date', 'close', 'open']).set_index('date')
    df['date'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df['close'] = pd.to_numeric(df['c'], errors='coerce')
    df['open'] = pd.to_numeric(df.get('o'), errors='coerce')
    return df.dropna(subset=['close']).set_index('date').sort_index()[['open', 'close']]
