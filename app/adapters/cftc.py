from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile
import re
import pandas as pd
from app.adapters.http import get_bytes, snapshot_payload

# CFTC historical compressed files. v0.3.0 uses TFF financial futures, futures-only.
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
                df = pd.read_csv(fh, low_memory=False)
            except Exception:
                fh.seek(0)
                df = pd.read_csv(fh, sep=',', low_memory=False)
    return df


def fetch_tff_years(start_year: int, end_year: int, raw_dir: Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(fetch_tff_year(year, raw_dir=raw_dir))
        except Exception as exc:  # keep partial annual pulls usable
            errors.append(f'{year}: {exc}')
    if not frames:
        raise RuntimeError('No CFTC TFF annual files could be fetched. ' + '; '.join(errors[:5]))
    out = pd.concat(frames, ignore_index=True, sort=False)
    out.attrs['fetch_errors'] = errors
    return out


def _norm_col(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_original = {_norm_col(c): c for c in df.columns}
    candidate_norms = [_norm_col(c) for c in candidates]
    for cand in candidate_norms:
        if cand in norm_to_original:
            return norm_to_original[cand]
    # loose contains fallback for CFTC naming variations
    for cand in candidate_norms:
        for norm, original in norm_to_original.items():
            if cand in norm:
                return original
    return None


def extract_jpy_tff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract leveraged-fund Japanese-yen positioning from CFTC TFF futures-only rows.

    Output is indexed by the CFTC as-of report date. Net values are normalised by open
    interest to make the signal more comparable through time.
    """
    market_col = _find_col(df, ['market_and_exchange_names', 'market and exchange names'])
    date_col = _find_col(df, ['report_date_as_yyyy-mm-dd', 'report_date_as_yyyy_mm_dd', 'as_of_date_in_form_yyyy-mm-dd'])
    oi_col = _find_col(df, ['open_interest_all', 'open interest all', 'open_interest'])
    long_col = _find_col(df, ['lev_money_positions_long_all', 'leveraged funds positions long all', 'leveraged funds-long all'])
    short_col = _find_col(df, ['lev_money_positions_short_all', 'leveraged funds positions short all', 'leveraged funds-short all'])
    spreading_col = _find_col(df, ['lev_money_positions_spread_all', 'leveraged funds positions spread all', 'leveraged funds-spreading all'])

    if not all([market_col, date_col, oi_col, long_col, short_col]):
        available = ', '.join(map(str, df.columns[:25]))
        raise RuntimeError('Could not identify required CFTC TFF JPY columns. First columns: ' + available)

    mask = df[market_col].astype(str).str.upper().str.contains('JAPANESE YEN', na=False)
    keep = [date_col, market_col, oi_col, long_col, short_col]
    if spreading_col:
        keep.append(spreading_col)
    out = df.loc[mask, keep].copy()
    rename = {date_col: 'date', market_col: 'market', oi_col: 'open_interest', long_col: 'long', short_col: 'short'}
    if spreading_col:
        rename[spreading_col] = 'spreading'
    out = out.rename(columns=rename)
    out['date'] = pd.to_datetime(out['date'], utc=True, errors='coerce')
    for c in ['open_interest', 'long', 'short', 'spreading']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['date', 'open_interest', 'long', 'short']).sort_values('date')
    out = out.drop_duplicates(subset=['date'], keep='last')
    out['lev_net_contracts'] = out['long'] - out['short']
    out['lev_net_oi'] = out['lev_net_contracts'] / out['open_interest'].replace(0, pd.NA)
    out['lev_short_oi'] = out['short'] / out['open_interest'].replace(0, pd.NA)
    out['lev_long_oi'] = out['long'] / out['open_interest'].replace(0, pd.NA)
    return out.set_index('date').sort_index()
