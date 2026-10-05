"""
Space weather data (CelesTrak SW-All.csv) for atmospheric models
Python conversion from yMATLAB/environment/loadSpaceWeather.m, lookupSolarGeoIndex.m
"""

import csv
import math
import os
import warnings

import numpy as np

from ..conversion import gc2jd
from ..time_utils import jd2gc


# Cache of the space weather data read from a file path (lookup_solar_geo_index)
_cached_sw = None
_cached_path = None


def load_space_weather(csv_path: str) -> dict:
    """
    # Read CelesTrak SW-All.csv

    Parameters
    ----------
    csv_path : str
        path to SW-All.csv

    Returns
    -------
    sw : dict
        - jdNoonStart: noon JD of the first row (integer)
        - jdNoonEnd: noon JD of the last row (integer)
        - nDays: number of rows (= consecutive days)
        - Kp: Kp of the 3-hour blocks, N x 8
        - Ap: Ap of the 3-hour blocks, N x 8
        - ApAvg: daily average Ap, N
        - F107obs: observed F10.7, N, sfu
        - F107obsCenter81: 81-day centered average of the observed F10.7, N, sfu

    Notes
    -----
    Data source: CelesTrak Space Weather file SW-All.csv
    https://celestrak.org/SpaceData/SW-All.csv
    Missing values (predicted rows) are NaN.

    Revisions
    ---------
    20260409  y.yoshimura

    See also
    --------
    lookup_solar_geo_index, jr1971
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Space weather file not found: {csv_path}')

    with open(csv_path, newline='') as fid:
        reader = csv.reader(fid)
        header = next(reader)
        rows = [row for row in reader if row]

    col = {name: i for i, name in enumerate(header)}

    def column(name):
        i = col[name]
        return np.array([float(row[i]) if row[i] != '' else np.nan for row in rows])

    sw = {}
    sw['nDays'] = len(rows)

    # First/last dates (yyyy-mm-dd) -> noon JD
    y0, m0, d0 = _parse_ymd(rows[0][col['DATE']])
    yN, mN, dN = _parse_ymd(rows[-1][col['DATE']])
    sw['jdNoonStart'] = float(gc2jd(y0, m0, d0, 12, 0, 0))
    sw['jdNoonEnd'] = float(gc2jd(yN, mN, dN, 12, 0, 0))

    # Consecutive days check
    expected_days = round(sw['jdNoonEnd'] - sw['jdNoonStart']) + 1
    if expected_days != sw['nDays']:
        warnings.warn(f"The number of rows of SW-All.csv ({sw['nDays']}) does not match "
                      f"the date span ({expected_days}).")

    # CelesTrak stores Kp as Kp*10 -> divide by 10
    sw['Kp'] = np.column_stack([column(f'KP{i}') for i in range(1, 9)]) / 10
    sw['Ap'] = np.column_stack([column(f'AP{i}') for i in range(1, 9)])
    sw['ApAvg'] = column('AP_AVG')

    sw['F107obs'] = column('F10.7_OBS')
    sw['F107obsCenter81'] = column('F10.7_OBS_CENTER81')

    return sw


# %[appendix]{"version":"1.0"}


def lookup_solar_geo_index(jd: float, sw) -> tuple:
    """
    # Solar and geomagnetic indices from CelesTrak SW-All.csv (for jr1971)

    Parameters
    ----------
    jd : float
        Julian day (UTC)
    sw : dict or str
        (a) dict read with load_space_weather() (recommended), or
        (b) path to SW-All.csv. The file is read once and cached; it is not
            read again as long as the same path is given.

    Returns
    -------
    F10 : float
        10.7 cm solar flux (1-day lag, observed), sfu
    F10a : float
        81-day centered average F10.7 (observed), sfu
    Kp : float
        Kp index (3-hour block at jd, no lag)

    Notes
    -----
    Data source: CelesTrak Space Weather file SW-All.csv
    https://celestrak.org/SpaceData/SW-All.csv

    Example:

        F10, F10a, Kp = lookup_solar_geo_index(gc2jd(2024, 1, 1, 12, 0, 0), 'SW-All.csv')

    Revisions
    ---------
    20260409  y.yoshimura

    See also
    --------
    load_space_weather, jr1971
    """
    global _cached_sw, _cached_path

    # Resolve the input: read from the cache when a path is given
    if isinstance(sw, (str, os.PathLike)):
        sw_path = os.fspath(sw)
        if _cached_sw is None or _cached_path != sw_path:
            _cached_sw = load_space_weather(sw_path)
            _cached_path = sw_path
        sw = _cached_sw
    elif not isinstance(sw, dict):
        raise TypeError('sw must be a dict from load_space_weather or a path to SW-All.csv.')

    jd = float(jd)

    # F10 / F10a (1-day lag, observed)
    # floor(jd - 0.5) is the noon JD (integer) of the previous UTC calendar day.
    jd_lag = math.floor(jd - 0.5)
    i_lag = round(jd_lag - sw['jdNoonStart'])
    if i_lag < 0 or i_lag >= sw['nDays']:
        raise ValueError(
            f"JD {jd:.3f} (lag date {_jd_to_date_str(jd_lag)}) is outside SW-All range "
            f"[{_jd_to_date_str(sw['jdNoonStart'])}, {_jd_to_date_str(sw['jdNoonEnd'])}].")
    F10 = float(sw['F107obs'][i_lag])
    F10a = float(sw['F107obsCenter81'][i_lag])

    if math.isnan(F10) or math.isnan(F10a):
        raise ValueError(
            f'F10/F10a is NaN at {_jd_to_date_str(jd_lag)} '
            '(likely beyond observed/centered-81 coverage).')

    # Kp (3-hour block at jd, no lag)
    # floor(jd + 0.5) is the noon JD (integer) of the UTC calendar day of jd.
    jd_today = math.floor(jd + 0.5)
    i_now = round(jd_today - sw['jdNoonStart'])
    if i_now < 0 or i_now >= sw['nDays']:
        raise ValueError(
            f"JD {jd:.3f} is outside SW-All range "
            f"[{_jd_to_date_str(sw['jdNoonStart'])}, {_jd_to_date_str(sw['jdNoonEnd'])}].")
    hr = jd2gc(jd)[3]
    block = int(hr // 3)  # 0..7
    Kp = float(sw['Kp'][i_now, block])

    return F10, F10a, Kp


# %[appendix]{"version":"1.0"}


def _parse_ymd(s: str) -> tuple:
    """Year, month, day from a 'yyyy-mm-dd' string"""
    y, m, d = (int(p) for p in s.split('-'))

    return y, m, d


def _jd_to_date_str(jd_noon: float) -> str:
    """Format an integer noon JD as 'yyyy-mm-dd'"""
    y, m, d = jd2gc(jd_noon)[:3]

    return f'{y:04d}-{m:02d}-{d:02d}'
