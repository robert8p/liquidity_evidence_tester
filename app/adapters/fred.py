from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

FRED_CSV_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'


def fetch_fred_csv(series_id: str, raw_dir: Path | None = None) -> pd.DataFrame:
    # Uses FRED's public chart CSV endpoint. This avoids requiring a FRED API key for v1.
    content = get_bytes(FRED_CSV_URL, params={'id': series_id})
    if raw_dir:
        snapshot_payload(raw_dir, f'fred_{series_id}', FRED_CSV_URL, content, suffix='csv')
    df = pd.read_csv(BytesIO(content))
    # FRED CSVs usually come as DATE,<SERIES_ID>
    date_col = df.columns[0]
    value_col = df.columns[1]
    df = df.rename(columns={date_col: 'date', value_col: 'value'})
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna(subset=['value']).set_index('date').sort_index()[['value']]
