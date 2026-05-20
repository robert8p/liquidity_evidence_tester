import pandas as pd
from app.release_calendar import h41_release_time, cftc_release_time, next_us_equity_open_after


def test_h41_release_after_wednesday_observation():
    observed = pd.Timestamp('2026-05-13 00:00:00Z')
    released = h41_release_time(observed)
    assert released == pd.Timestamp('2026-05-14 20:30:00Z')
    assert released > observed


def test_cftc_release_after_tuesday_observation():
    observed = pd.Timestamp('2026-05-12 00:00:00Z')
    released = cftc_release_time(observed)
    assert released == pd.Timestamp('2026-05-15 19:30:00Z')
    assert released > observed


def test_equity_open_rolls_after_close_to_next_day():
    ts = pd.Timestamp('2026-05-14 20:30:00Z')  # 16:30 ET after cash close
    nxt = next_us_equity_open_after(ts)
    assert nxt == pd.Timestamp('2026-05-15 13:30:00Z')


def test_equity_open_rolls_weekend():
    ts = pd.Timestamp('2026-05-15 20:30:00Z')
    nxt = next_us_equity_open_after(ts)
    assert nxt.weekday() == 0
