"""
TLE (Two-Line Element) reading and processing
Python conversion from yMATLAB/orbit/readTLE.m
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class TLEData:
    """TLE data container"""
    sat_name: str = ""
    sat_id: str = ""
    launch_year: int = 0
    launch_num: int = 0
    launch_piece: str = ""
    oe: np.ndarray = field(default_factory=lambda: np.zeros(6))  # [a, e, i, RAAN, w, M]
    jd: float = 0.0  # Julian day of TT
    n_rev: float = 0.0  # mean motion [rev/day]


def read_tle(tle_filename: str, const, tool: str = 'yoshimuLibrary') -> List[TLEData]:
    """
    # Reading TLE (Two-Line Elements)

    Parameters
    ----------
    tle_filename : str
        Two line elements file name
    const : OrbitalConstants
        constant parameters for orbit propagation
    tool : str
        'yoshimuLibrary' (default), 'SPICE', or 'MATLAB'

    Returns
    -------
    tle_list : list of TLEData
        list of TLE data objects containing:
        - sat_name: satellite name
        - sat_id: satellite ID
        - launch_year: launch year
        - launch_num: launch number
        - launch_piece: launch piece
        - oe: mean orbital elements [a, e, i, RAAN, w, M] (km, rad)
        - jd: Julian day of Terrestrial Time
        - n_rev: mean motion [rev/day]

    Notes
    -----
    TLEを読み込み

    References
    ----------
    NA

    Revisions
    ---------
    20190201  y.yoshimura

    See also
    --------
    orbit_const
    """
    from ..time_utils import leap_s, dat, utc2tt, ut2tt
    from ..conversion import gc2jd, s2day
    from ..time_utils import jd2mjd

    # Read TLE file
    with open(tle_filename, 'r') as f:
        lines = f.readlines()

    # Remove empty lines and strip whitespace
    lines = [line.strip() for line in lines if line.strip()]

    n_lines = len(lines)

    # Determine format (with or without satellite name)
    if lines[0][0] == '1':
        # No satellite name
        has_name = False
        n_entries = n_lines // 2
    else:
        # Has satellite name
        has_name = True
        n_entries = n_lines // 3

    tle_list = []

    for entry_idx in range(n_entries):
        tle = TLEData()

        if has_name:
            line0 = lines[entry_idx * 3]
            line1 = lines[entry_idx * 3 + 1]
            line2 = lines[entry_idx * 3 + 2]
            tle.sat_name = line0.strip()
        else:
            line1 = lines[entry_idx * 2]
            line2 = lines[entry_idx * 2 + 1]

        # Parse LINE1
        tle.sat_id = line1[2:7].strip()
        try:
            tle.launch_year = int(line1[9:11])
            tle.launch_num = int(line1[11:14])
            tle.launch_piece = line1[14:17].strip()
        except ValueError:
            pass

        # Epoch
        try:
            year = int(line1[18:20])
            epoch_day = float(line1[20:32])
        except ValueError:
            year = 0
            epoch_day = 0.0

        # Convert 2-digit year to 4-digit
        if year > 57:
            epoch_year = 1900 + year
        else:
            epoch_year = 2000 + year

        # Calculate Julian day
        # epoch_day is day of year (with fractional day)
        day_of_year = int(epoch_day)
        frac_day = epoch_day - day_of_year

        # Convert day of year to month and day
        import datetime
        date = datetime.datetime(epoch_year, 1, 1) + datetime.timedelta(days=day_of_year - 1 + frac_day)
        jd = gc2jd(date.year, date.month, date.day, date.hour, date.minute, date.second + date.microsecond / 1e6)

        # Convert to TT
        if tool == 'yoshimuLibrary':
            leap_jd = leap_s()
            delta_at = dat(jd, leap_jd)
            jd = utc2tt(jd, delta_at)
        elif tool == 'MATLAB':
            mjd = jd2mjd(jd)
            # Approximate conversion
            jd = jd + s2day(ut2tt(jd))
        else:
            # Default: UT1 to TT approximation
            jd = jd + s2day(ut2tt(jd))

        tle.jd = jd

        # Parse LINE2
        try:
            rpd = float(line2[52:63])  # revolutions per day
            tle.n_rev = rpd

            # Orbital elements
            # Semi-major axis from mean motion
            n_rad_s = 2 * np.pi * rpd / 86400  # rad/s
            a = (const.GE / n_rad_s**2)**(1/3)  # km

            # Eccentricity (assumed decimal point)
            e_str = '0.' + line2[26:33].strip()
            e = float(e_str)

            # Inclination (deg)
            inc = float(line2[8:16])

            # RAAN (deg)
            raan = float(line2[17:25])

            # Argument of perigee (deg)
            w = float(line2[34:42])

            # Mean anomaly (deg)
            M = float(line2[43:51])

            # Convert to radians and wrap
            inc = np.mod(inc, 360) * np.pi / 180
            raan = np.mod(raan, 360) * np.pi / 180
            w = np.mod(w, 360) * np.pi / 180
            M = np.mod(M, 360) * np.pi / 180

            tle.oe = np.array([a, e, inc, raan, w, M])

        except (ValueError, IndexError):
            pass

        tle_list.append(tle)

    return tle_list


def read_tle_single(tle_filename: str, const, tool: str = 'yoshimuLibrary') -> TLEData:
    """
    # Read a single TLE from file

    Parameters
    ----------
    tle_filename : str
        TLE file name
    const : OrbitalConstants
        orbital constants
    tool : str
        time conversion tool

    Returns
    -------
    tle : TLEData
        TLE data
    """
    tle_list = read_tle(tle_filename, const, tool)
    if tle_list:
        return tle_list[0]
    return TLEData()


# %[appendix]{"version":"1.0"}
