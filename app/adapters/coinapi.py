from __future__ import annotations

from pathlib import Path
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

BASE_URL = 'https://rest.coinapi.io/v1/ohlcv/{symbol_id}/history'


def fetch_ohlcv(symbol_id: str, api_key: str, *, start_iso: str, end_iso: str | None = None, period_id: str = '1DAY', raw_dir: Path | None = None) -> pd.DataFrame:
    headers = {'X-CoinAPI-Key': api_key}
    params: dict[str, str] = {'period_id': period_id, 'time_start': start_iso, 'limit': '100000'}
    if end_iso:
        params['time_end'] = end_iso
    url = BASE_URL.format(symbol_id=symbol_id)
    content = get_bytes(url, headers=headers, params=params)
    if raw_dir:
        snapshot_payload(raw_dir, f'coinapi_{symbol_id}', url, content, suffix='json')
    df = pd.read_json(content)
    if df.empty:
        return pd.DataFrame(columns=['date', 'close']).set_index('date')
    time_col = 'time_period_start'
    close_col = 'price_close'
    df['date'] = pd.to_datetime(df[time_col], utc=True)
    df['close'] = pd.to_numeric(df[close_col], errors='coerce')
    return df.dropna(subset=['close']).set_index('date').sort_index()[['close']]
