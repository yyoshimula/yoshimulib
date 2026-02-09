"""
Sidereal time calculations
Python conversion from yMATLAB/orbit/gmst.m, gast.m, era.m
"""

import numpy as np


def gmst(jd: np.ndarray) -> np.ndarray:
    """
    # Julian day to Greenwich mean sidereal time

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, n x 1 vector

    Returns
    -------
    GMST : np.ndarray
        Greenwich mean sidereal time, rad, n x 1 vector

    Notes
    -----
    NA

    References
    ----------
    Jean Meeus, "Astronomical Algorithms, 2nd edition", p.87. Eq.(12.3) or (12.4)

    Revisions
    ---------
    20160419  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gast
    """
    jd = np.atleast_1d(jd)
    T = (jd - 2451545.0) / 36525.0

    # Eq.(12.4)
    GMST = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T ** 2 - T ** 3 / 38710000.0)

    GMST = np.deg2rad(np.mod(GMST, 360))

    return GMST


def gast(jd: np.ndarray, const) -> np.ndarray:
    """
    # Greenwich apparent sidereal time (GAST) from Julian day

    Calculation based on IAU-76/FK5

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, n x 1 vector
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    GAST : np.ndarray
        Greenwich apparent sidereal time, rad, n x 1 vector

    Notes
    -----
    NA

    References
    ----------
    Jean Meeus, "Astronomical Algorithms, 2nd edition", p.87. Eq.(12.3) or (12.4)

    Revisions
    ---------
    20160419  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gmst
    """
    from .precession import nutation, obliquity

    GMST_val = gmst(jd)  # Greenwich mean sidereal time
    mean_epsi = obliquity(jd, const)  # mean obliquity of the ecliptic, rad

    dpsi, deps, _, _ = nutation(jd, const)

    GAST_val = GMST_val + dpsi * np.cos(mean_epsi + deps)

    return GAST_val


def era(jd_ut1: np.ndarray) -> np.ndarray:
    """
    # Earth rotation angle (IAU-2006/2000, CIO-based)

    Parameters
    ----------
    jd_ut1 : np.ndarray
        Julian day of UT1, day

    Returns
    -------
    theta : np.ndarray
        Earth rotation angle, rad

    Notes
    -----
    NA

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics and
    Applications. Springer Science & Business Media. 4th edition, p.212

    Revisions
    ---------
    20230605  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    leap_s
    """
    jd_ut1 = np.atleast_1d(jd_ut1)

    theta = 0.779057273264 + 1.00273781191135448 * (jd_ut1 - 2451545.0)
    theta = np.mod(theta * 2.0 * np.pi, 2 * np.pi)

    return theta


# %[appendix]{"version":"1.0"}
