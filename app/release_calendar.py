from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

NY = ZoneInfo('America/New_York')
UTC = timezone.utc


def _to_ny_date(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert(NY).to_pydatetime()


def h41_release_time(observed_at_utc: pd.Timestamp) -> pd.Timestamp:
    """H.4.1 values are treated as Wednesday levels released Thursday 16:30 ET.

    FRED stores only dates, so this maps each observed date to the following Thursday release.
    If a source date is already Thursday, the same Thursday 16:30 ET is used.
    """
    dt_ny = _to_ny_date(observed_at_utc)
    # Weekday: Monday=0, Wednesday=2, Thursday=3
    days_to_thursday = (3 - dt_ny.weekday()) % 7
    release_date = (dt_ny + timedelta(days=days_to_thursday)).date()
    release_dt = datetime.combine(release_date, time(16, 30), tzinfo=NY)
    return pd.Timestamp(release_dt.astimezone(UTC))


def cftc_release_time(observed_at_utc: pd.Timestamp) -> pd.Timestamp:
    """CFTC COT/TFF positions are Tuesday observations released Friday 15:30 ET."""
    dt_ny = _to_ny_date(observed_at_utc)
    days_to_friday = (4 - dt_ny.weekday()) % 7
    release_date = (dt_ny + timedelta(days=days_to_friday)).date()
    release_dt = datetime.combine(release_date, time(15, 30), tzinfo=NY)
    return pd.Timestamp(release_dt.astimezone(UTC))


def next_us_equity_open_after(ts_utc: pd.Timestamp) -> pd.Timestamp:
    """Simple conservative weekday-only US equity regular-session open.

    This is intentionally conservative and does not model exchange holidays in v1.
    Later versions can replace this with an exchange calendar.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize('UTC')
    dt_ny = ts_utc.tz_convert(NY).to_pydatetime()
    open_dt = datetime.combine(dt_ny.date(), time(9, 30), tzinfo=NY)
    if dt_ny >= open_dt:
        open_dt += timedelta(days=1)
    while open_dt.weekday() >= 5:
        open_dt += timedelta(days=1)
    return pd.Timestamp(open_dt.astimezone(UTC))


def next_utc_midnight_after(ts_utc: pd.Timestamp) -> pd.Timestamp:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize('UTC')
    dt = ts_utc.tz_convert('UTC').to_pydatetime()
    nxt = datetime(dt.year, dt.month, dt.day, tzinfo=UTC) + timedelta(days=1)
    return pd.Timestamp(nxt)


def attach_h41_alignment(df: pd.DataFrame, instrument_type: str) -> pd.DataFrame:
    out = df.copy()
    out['observed_at_utc'] = out.index
    out['released_at_utc'] = out['observed_at_utc'].map(h41_release_time)
    if instrument_type == 'equity':
        out['effective_trade_at_utc'] = out['released_at_utc'].map(next_us_equity_open_after)
    elif instrument_type == 'crypto':
        out['effective_trade_at_utc'] = out['released_at_utc']
    else:
        out['effective_trade_at_utc'] = out['released_at_utc'].map(next_utc_midnight_after)
    return out
