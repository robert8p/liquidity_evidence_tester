from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

# Public Cboe CDN paths. If Cboe changes these, set overrides in code/config.
VIX_HISTORY_URL = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
VVIX_HISTORY_URL = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv'


def fetch_cboe_history(url: str, raw_dir: Path | None = None, source: str = 'cboe') -> pd.DataFrame:
    content = get_bytes(url)
    if raw_dir:
        snapshot_payload(raw_dir, source, url, content, suffix='csv')
    df = pd.read_csv(BytesIO(content))
    # Cboe CSVs vary in exact casing. Use resilient matching.
    lower = {c.lower(): c for c in df.columns}
    date_col = lower.get('date') or df.columns[0]
    close_col = lower.get('close') or lower.get('vix close') or lower.get('vvix close') or df.columns[-1]
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], utc=True), 'close': pd.to_numeric(df[close_col], errors='coerce')})
    return out.dropna(subset=['close']).set_index('date').sort_index()[['close']]
