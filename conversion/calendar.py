"""
Calendar conversion functions (Gregorian to Julian Day)
Python conversion from yMATLAB/conversion/
"""

import numpy as np


def gc2jd(year: int | np.ndarray,
          month: int | np.ndarray,
          day: int | np.ndarray,
          hour: int | np.ndarray = 0,
          minute: int | np.ndarray = 0,
          second: float | np.ndarray = 0.0) -> float | np.ndarray:
    """
    # converting Gregorian calendar date to Julian days

    Parameters
    ----------
    year : int or np.ndarray
        year, nx1 vector
    month : int or np.ndarray
        month, nx1 vector
    day : int or np.ndarray
        day, nx1 vector
    hour : int or np.ndarray, optional
        hour, nx1 vector (default: 0)
    minute : int or np.ndarray, optional
        minute, nx1 vector (default: 0)
    second : float or np.ndarray, optional
        second, nx1 vector (default: 0.0)

    Returns
    -------
    jd : float or np.ndarray
        Julian day, nx1 vector

    Notes
    -----
    NA

    References
    ----------
    confirmed with http://eco.mtk.nao.ac.jp/cgi-bin/koyomi/cande/date2jd.cgi
    Meeus, J., Astronomical Algorithms, 1998., p61, Eq.(7.1)

    Revisions
    ---------
    20160630  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    jd2gc
    """
    # Convert to numpy arrays if needed for element-wise operations
    year = np.asarray(year)
    month = np.asarray(month)
    day = np.asarray(day)
    hour = np.asarray(hour)
    minute = np.asarray(minute)
    second = np.asarray(second)

    # Adjust month and year for January and February
    M_ = np.where(month > 2, month, month + 12)
    Y_ = np.where(month > 2, year, year - 1)

    A_ = np.floor(Y_ / 100)
    B_ = 2 - A_ + np.floor(A_ / 4)

    jd = (np.floor(365.25 * (Y_ + 4716)) + np.floor(30.6001 * (M_ + 1))
          + day + B_ - 1524.5 + hour / 24 + minute / 1440 + second / 86400)

    # Return scalar if input was scalar
    if jd.ndim == 0:
        return float(jd)
    return jd


# %[appendix]{"version":"1.0"}
