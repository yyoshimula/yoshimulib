"""
Julian day related conversion functions
Python conversion from yMATLAB/time/
"""

import numpy as np


def jd2gc(jd: float | np.ndarray) -> tuple:
    """
    # calculate the Gregorian calendar date from the Julian date

    Parameters
    ----------
    jd : float or np.ndarray
        Julian day, day

    Returns
    -------
    year : int or np.ndarray
        year
    month : int or np.ndarray
        month
    day : int or np.ndarray
        day
    hour : int or np.ndarray
        hour
    minute : int or np.ndarray
        minute
    second : float or np.ndarray
        second

    Notes
    -----
    checked with: http://eco.mtk.nao.ac.jp/cgi-bin/koyomi/cande/jd2date.cgi

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    jd2fyear, gc2jd
    """
    jd = np.asarray(jd)
    scalar_input = jd.ndim == 0
    jd = np.atleast_1d(jd)

    fday = jd + 0.5 - np.floor(jd + 0.5)
    jd_int = np.floor(jd + 0.5)
    in2 = np.floor(np.floor((jd_int - 4479.5) / 36524.25) * 0.75 + 0.5) - 37
    in2 = jd_int + in2

    year = np.floor(in2 / 365.25) - 4712
    an2 = in2 - 59.25
    an2 = np.floor(an2 - np.floor(an2 / 365.25) * 365.25) + 0.5
    mon2 = np.floor(an2 / 30.6) + 2

    month = mon2 - np.floor(mon2 / 12) * 12 + 1
    day = np.floor(an2 - np.floor(an2 / 30.6) * 30.6 + 1)
    hour = np.floor(fday * 24)
    minute = np.floor((fday * 24 - np.floor(fday * 24)) * 60)
    second = (fday * 24.0 * 60.0 * 60.0 - np.floor(fday * 24) * 60.0 * 60.0
              - np.floor((fday * 24 - np.floor(fday * 24)) * 60) * 60.0)

    # Convert to int where appropriate
    year = year.astype(int)
    month = month.astype(int)
    day = day.astype(int)
    hour = hour.astype(int)
    minute = minute.astype(int)

    if scalar_input:
        return int(year[0]), int(month[0]), int(day[0]), int(hour[0]), int(minute[0]), float(second[0])

    return year, month, day, hour, minute, second


# %[appendix]{"version":"1.0"}


def jd2jdt(jd: float | np.ndarray) -> float | np.ndarray:
    """
    # Julian century (Julian Day to Julian Century)

    Parameters
    ----------
    jd : float or np.ndarray
        Julian day

    Returns
    -------
    T : float or np.ndarray
        Julian century since J2000.0

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20230605  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    leaps
    """
    T = (jd - 2451545.0) / 36525.0

    return T


# %[appendix]{"version":"1.0"}


def jd2mjd(jd: float | np.ndarray) -> float | np.ndarray:
    """
    # Julian date to modified Julian date

    Parameters
    ----------
    jd : float or np.ndarray
        Julian day, day

    Returns
    -------
    mjd : float or np.ndarray
        modified Julian day, day

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20230202  y.yoshimura

    See also
    --------
    mjd2jd
    """
    mjd = jd - 2400000.5

    return mjd


# %[appendix]{"version":"1.0"}


def mjd2jd(mjd: float | np.ndarray) -> float | np.ndarray:
    """
    # modified Julian date to Julian date

    Parameters
    ----------
    mjd : float or np.ndarray
        modified Julian day, day

    Returns
    -------
    jd : float or np.ndarray
        Julian day, day

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20230202  y.yoshimura

    See also
    --------
    jd2mjd
    """
    jd = mjd + 2400000.5

    return jd


# %[appendix]{"version":"1.0"}
