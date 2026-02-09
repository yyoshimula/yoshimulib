"""
Unit conversion functions
Python conversion from yMATLAB/conversion/
"""

import numpy as np


def au2km(au: float | np.ndarray, au_const: float = 149597870.7) -> float | np.ndarray:
    """
    # converting astronomical unit (AU) to km

    Parameters
    ----------
    au : float or np.ndarray
        distance to be converted, AU
    au_const : float, optional
        AU constant in km (default: 149597870.7 km, IAU 2012)

    Returns
    -------
    km : float or np.ndarray
        distance converted, km

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    km2au
    """
    km = au * au_const

    return km


# %[appendix]{"version":"1.0"}


def km2au(km: float | np.ndarray, au_const: float = 149597870.7) -> float | np.ndarray:
    """
    # converting km to astronomical unit

    Parameters
    ----------
    km : float or np.ndarray
        distance to be converted, km
    au_const : float, optional
        AU constant in km (default: 149597870.7 km, IAU 2012)

    Returns
    -------
    au : float or np.ndarray
        distance converted, AU

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    au2km
    """
    au = km / au_const

    return au


# %[appendix]{"version":"1.0"}


def rad2arcs(angle: float | np.ndarray) -> float | np.ndarray:
    """
    # converting radian to arcsecond

    Parameters
    ----------
    angle : float or np.ndarray
        angle to be converted, rad

    Returns
    -------
    arcs : float or np.ndarray
        angle converted, arcseconds

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    arcs2rad
    """
    arcs = angle * 3600.0 * 180.0 / np.pi

    return arcs


# %[appendix]{"version":"1.0"}


def arcs2rad(angle: float | np.ndarray) -> float | np.ndarray:
    """
    # converting arcsecond to radian

    Parameters
    ----------
    angle : float or np.ndarray
        angle to be converted, arcseconds

    Returns
    -------
    rad : float or np.ndarray
        angle converted, rad

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    rad2arcs
    """
    rad = angle * np.pi / 3600.0 / 180.0

    return rad


# %[appendix]{"version":"1.0"}


def hms2deg(hour: float | np.ndarray,
            minute: float | np.ndarray,
            sec: float | np.ndarray) -> float | np.ndarray:
    """
    # hour, min, and sec angles to deg

    Parameters
    ----------
    hour : float or np.ndarray
        hour angle
    minute : float or np.ndarray
        minute angle
    sec : float or np.ndarray
        second angle

    Returns
    -------
    out : float or np.ndarray
        angle, deg

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20210419  y.yoshimura

    See also
    --------
    NA
    """
    out = hour * 15.0 + (minute * 60 + sec) / 3600.0

    return out


# %[appendix]{"version":"1.0"}


def s2day(s: float | np.ndarray) -> float | np.ndarray:
    """
    # transform second to days

    Parameters
    ----------
    s : float or np.ndarray
        time in seconds

    Returns
    -------
    day : float or np.ndarray
        time in days

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20210215  y.yoshimura

    See also
    --------
    NA
    """
    day = s / 24.0 / 60.0 / 60.0

    return day


# %[appendix]{"version":"1.0"}
