from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

BASE_URL = 'https://data.alpaca.markets/v2/stocks/bars'


def fetch_daily_bars(symbol: str, key_id: str, secret_key: str, start_iso: str, end_iso: str, raw_dir: Path | None = None, feed: str = 'iex') -> pd.DataFrame:
    headers = {'APCA-API-KEY-ID': key_id, 'APCA-API-SECRET-KEY': secret_key}
    params = {
        'symbols': symbol,
        'timeframe': '1Day',
        'start': start_iso,
        'end': end_iso,
        'adjustment': 'all',
        'feed': feed,
        'limit': '10000',
    }
    content = get_bytes(BASE_URL, headers=headers, params=params)
    if raw_dir:
        snapshot_payload(raw_dir, f'alpaca_{symbol}', BASE_URL, content, suffix='json')
    payload = json.loads(content.decode('utf-8'))
    bars = (payload.get('bars') or {}).get(symbol, [])
    df = pd.DataFrame(bars)
    if df.empty:
        return pd.DataFrame(columns=['date', 'close', 'open']).set_index('date')
    df['date'] = pd.to_datetime(df['t'], utc=True)
    df['open'] = pd.to_numeric(df.get('o'), errors='coerce')
    df['close'] = pd.to_numeric(df['c'], errors='coerce')
    return df.dropna(subset=['close']).set_index('date').sort_index()[['open', 'close']]
