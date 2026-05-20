from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

# TFF annual text files are published by CFTC. Users can override URL/year in future versions.
TFF_FINANCIAL_FUTURES_URL_TEMPLATE = 'https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip'


def fetch_tff_year(year: int, raw_dir: Path | None = None) -> pd.DataFrame:
    url = TFF_FINANCIAL_FUTURES_URL_TEMPLATE.format(year=year)
    content = get_bytes(url)
    if raw_dir:
        snapshot_payload(raw_dir, f'cftc_tff_{year}', url, content, suffix='zip')
    with zipfile.ZipFile(BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(('.txt', '.csv'))]
        if not names:
            raise RuntimeError('No TXT/CSV file found inside CFTC zip')
        with zf.open(names[0]) as fh:
            try:
                df = pd.read_csv(fh)
            except Exception:
                fh.seek(0)
                df = pd.read_csv(fh, low_memory=False)
    return df


def extract_jpy_tff_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    market_col = cols.get('market_and_exchange_names') or cols.get('market and exchange names') or cols.get('market_and_exchange_name')
    date_col = cols.get('report_date_as_yyyy-mm-dd') or cols.get('report_date_as_yyyy_mm_dd') or cols.get('as_of_date_in_form_yyyy-mm-dd')
    oi_col = cols.get('open_interest_all') or cols.get('open_interest')
    long_col = cols.get('lev_money_positions_long_all')
    short_col = cols.get('lev_money_positions_short_all')
    if not all([market_col, date_col, oi_col, long_col, short_col]):
        raise RuntimeError('Could not identify required CFTC TFF columns for leveraged-fund JPY extraction')
    mask = df[market_col].astype(str).str.upper().str.contains('JAPANESE YEN', na=False)
    out = df.loc[mask, [date_col, oi_col, long_col, short_col]].copy()
    out = out.rename(columns={date_col: 'date', oi_col: 'open_interest', long_col: 'long', short_col: 'short'})
    out['date'] = pd.to_datetime(out['date'], utc=True)
    for c in ['open_interest', 'long', 'short']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna().sort_values('date')
    out['net_oi'] = (out['long'] - out['short']) / out['open_interest'].replace(0, pd.NA)
    return out.set_index('date')
