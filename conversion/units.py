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
    au_const : float or OrbitalConstants, optional
        AU constant in km (default: 149597870.7 km, IAU 2012), or the orbital
        constants (as in MATLAB, where const.AU is used)

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
    km = au * getattr(au_const, 'AU', au_const)

    return km


# %[appendix]{"version":"1.0"}


def km2au(km: float | np.ndarray, au_const: float = 149597870.7) -> float | np.ndarray:
    """
    # converting km to astronomical unit

    Parameters
    ----------
    km : float or np.ndarray
        distance to be converted, km
    au_const : float or OrbitalConstants, optional
        AU constant in km (default: 149597870.7 km, IAU 2012), or the orbital
        constants (as in MATLAB, where const.AU is used)

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
    au = km / getattr(au_const, 'AU', au_const)

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
    # degree, arcminute, arcsecond to degree (DMS -> deg)

    Parameters
    ----------
    hour : float or np.ndarray
        degree part, deg
    minute : float or np.ndarray
        arcminute, arcmin
    sec : float or np.ndarray
        arcsecond, arcsec

    Returns
    -------
    out : float or np.ndarray
        angle, deg

    Notes
    -----
    deg = hour + min/60 + sec/3600 (not the x15 conversion of right-ascension
    hours), consistent with the mean obliquity 23 deg 26' 21.448".

    References
    ----------
    NA

    Revisions
    ---------
    20210419  y.yoshimura
    20260707  y.yoshimura, removed the wrong x15 on the hour term (DMS -> deg)

    See also
    --------
    obliquity
    """
    out = hour + (minute * 60 + sec) / 3600.0

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


def day2s(day: float | np.ndarray) -> float | np.ndarray:
    """
    # transform days to seconds

    Parameters
    ----------
    day : float or np.ndarray
        time in days

    Returns
    -------
    s : float or np.ndarray
        time in seconds

    Notes
    -----
    Inverse of s2day.

    References
    ----------
    NA

    Revisions
    ---------
    20260707  y.yoshimura

    See also
    --------
    s2day
    """
    s = day * 24.0 * 60.0 * 60.0

    return s


# %[appendix]{"version":"1.0"}
