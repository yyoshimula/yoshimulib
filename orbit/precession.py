"""
Precession and nutation calculations
Python conversion from yMATLAB/orbit/precession.m, nutation.m, etc.
"""

import numpy as np
from .constants import arcs2rad, rad2arcs


def precession(jd0: np.ndarray, jd1: np.ndarray, const) -> tuple:
    """
    # Precession angles from mean to TOD

    Parameters
    ----------
    jd0 : np.ndarray
        Julian day, day, scalar or array
    jd1 : np.ndarray
        Julian day, day, scalar or array
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    zeta : np.ndarray
        precession angle, rad
    z : np.ndarray
        precession angle, rad
    theta : np.ndarray
        precession angle, rad
    eta : np.ndarray
        angle of inclination between the two ecliptics, rad
    Pi_ : np.ndarray
        angle from initial equinox to intersection of ecliptics, rad
    p : np.ndarray
        combined precession in longitude, rad

    Notes
    -----
    NA

    References
    ----------
    [1] Montenbruck, O., and Gill, E., Satellite Orbits, Berlin, Heidelberg:
        Springer Science & Business Media, 2012. p.176
    [2] Jean Meeus, "Astronomical Algorithms, 2nd Edition", pp.134-138

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    nutation_q, precession_dcm
    """
    jd0 = np.atleast_1d(jd0)
    jd1 = np.atleast_1d(jd1)

    T = (jd0 - const.J2000) / 36525.0
    t = (jd1 - jd0) / 36525.0

    zeta = ((2306.2181 + 1.39656 * T - 0.000139 * T ** 2) * t
            + (0.30188 - 0.000344 * T) * t ** 2
            + 0.017998 * t ** 3)
    zeta = arcs2rad(zeta)

    z = ((2306.2181 + 1.39656 * T - 0.000139 * T ** 2) * t
         + (1.09468 + 0.000066 * T) * t ** 2
         + 0.018203 * t ** 3)
    z = arcs2rad(z)

    theta = ((2004.3109 - 0.85330 * T - 0.000217 * T ** 2) * t
             - (0.42665 + 0.000217 * T) * t ** 2
             - 0.041833 * t ** 3)
    theta = arcs2rad(theta)

    # angle of inclination between the two ecliptics. Eq. (21.5) [ref.2]
    eta = ((47.0029 - 0.06603 * T + 0.000598 * T ** 2) * t
           + (-0.03302 + 0.000598 * T) * t ** 2
           + 0.000060 * t ** 3)
    eta = arcs2rad(eta)

    # initial equinox to intersection of ecliptics, Eq. (21.5) [ref.2]
    Pi_ = (3289.4789 * T + 0.60622 * T ** 2
           - (869.8089 + 0.50491 * T) * t
           + 0.03536 * t ** 2)
    Pi_ = np.deg2rad(174.876384) + arcs2rad(Pi_)

    # combined precession in longitude, rad, Eq. (21.5) [ref.2]
    p = ((5029.0966 + 2.222226 * T - 0.0000042 * T ** 2) * t
         + (1.11113 - 0.000042 * T) * t ** 2
         - 0.000006 * t ** 3)
    p = arcs2rad(p)

    return zeta, z, theta, eta, Pi_, p


def precession_dcm(jd0: np.ndarray, jd1: np.ndarray, const) -> np.ndarray:
    """
    # Precession rotation matrix (DCM)

    Parameters
    ----------
    jd0 : np.ndarray
        Julian day, day
    jd1 : np.ndarray
        Julian day, day
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    P : np.ndarray
        Precession rotation matrix, 3x3

    References
    ----------
    Montenbruck, O., and Gill, E., Satellite Orbits, p.176

    Revisions
    ---------
    20211027  y.yoshimura
    """
    zeta, z, theta, _, _, _ = precession(jd0, jd1, const)

    # Precession matrix
    cz = np.cos(zeta)
    sz = np.sin(zeta)
    czz = np.cos(z)
    szz = np.sin(z)
    ct = np.cos(theta)
    st = np.sin(theta)

    P = np.array([
        [cz * ct * czz - sz * szz, -sz * ct * czz - cz * szz, -st * czz],
        [cz * ct * szz + sz * czz, -sz * ct * szz + cz * czz, -st * szz],
        [cz * st, -sz * st, ct]
    ])

    return P


def precession_q(jd0: np.ndarray, jd1: np.ndarray, const) -> np.ndarray:
    """
    # Precession quaternion

    Parameters
    ----------
    jd0 : np.ndarray
        Julian day, day
    jd1 : np.ndarray
        Julian day, day
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    q : np.ndarray
        Precession quaternion [q1, q2, q3, q4] (scalar last)

    References
    ----------
    Montenbruck, O., and Gill, E., Satellite Orbits

    Revisions
    ---------
    20211027  y.yoshimura
    """
    from ..attitude import zxz2q

    zeta, z, theta, _, _, _ = precession(jd0, jd1, const)

    # q = R3(-zeta) * R2(theta) * R3(-z) via z-x-z convention
    # Actually the rotation is R3(-z) * R2(theta) * R3(-zeta)
    # Using ZXZ Euler angles:
    q = zxz2q(-zeta, theta, -z, scalar=4)

    return q


def nutation(jd: np.ndarray, const) -> tuple:
    """
    # Calculate nutation angles (dpsi, deps)

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    dpsi : np.ndarray
        nutation in longitude, rad
    deps : np.ndarray
        nutation in obliquity, rad
    eps : np.ndarray
        true obliquity, rad
    eps0 : np.ndarray
        mean obliquity, rad

    References
    ----------
    Montenbruck, O., and Gill, E., Satellite Orbits, p.78

    Revisions
    ---------
    20211027  y.yoshimura
    """
    jd = np.atleast_1d(jd)

    T = (jd - const.J2000) / 36525.0

    # Mean obliquity of the ecliptic, arcsec
    eps0_arcs = 84381.448 - 46.8150 * T - 0.00059 * T ** 2 + 0.001813 * T ** 3
    eps0 = arcs2rad(eps0_arcs)

    # Fundamental arguments, deg
    # Mean longitude of Moon's ascending node
    Om = 125.04452 - 1934.136261 * T + 0.0020708 * T ** 2 + T ** 3 / 450000
    Om = np.deg2rad(np.mod(Om, 360))

    # Mean longitude of Sun
    L_s = 280.4665 + 36000.7698 * T
    L_s = np.deg2rad(np.mod(L_s, 360))

    # Mean longitude of Moon
    L_m = 218.3165 + 481267.8813 * T
    L_m = np.deg2rad(np.mod(L_m, 360))

    # Nutation in longitude and obliquity (IAU 1980 model - simplified)
    dpsi = (-17.20 * np.sin(Om) - 1.32 * np.sin(2 * L_s)
            - 0.23 * np.sin(2 * L_m) + 0.21 * np.sin(2 * Om))
    dpsi = arcs2rad(dpsi)

    deps = (9.20 * np.cos(Om) + 0.57 * np.cos(L_s)
            + 0.10 * np.cos(L_m) - 0.09 * np.cos(2 * Om))
    deps = arcs2rad(deps)

    # True obliquity
    eps = eps0 + deps

    return dpsi, deps, eps, eps0


def nutation_dcm(jd: np.ndarray, const) -> np.ndarray:
    """
    # Nutation rotation matrix (DCM)

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    N : np.ndarray
        Nutation rotation matrix, 3x3

    References
    ----------
    Montenbruck, O., and Gill, E., Satellite Orbits

    Revisions
    ---------
    20211027  y.yoshimura
    """
    dpsi, deps, eps, eps0 = nutation(jd, const)

    # Nutation matrix
    ce0 = np.cos(eps0)
    se0 = np.sin(eps0)
    ce = np.cos(eps)
    se = np.sin(eps)
    cp = np.cos(dpsi)
    sp = np.sin(dpsi)

    N = np.array([
        [cp, -sp * ce0, -sp * se0],
        [sp * ce, cp * ce0 * ce + se0 * se, cp * se0 * ce - ce0 * se],
        [sp * se, cp * ce0 * se - se0 * ce, cp * se0 * se + ce0 * ce]
    ])

    return N


def nutation_q(jd: np.ndarray, const) -> np.ndarray:
    """
    # Nutation quaternion

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    q : np.ndarray
        Nutation quaternion [q1, q2, q3, q4] (scalar last)

    Revisions
    ---------
    20211027  y.yoshimura
    """
    from ..attitude import zxz2q

    dpsi, deps, eps, eps0 = nutation(jd, const)

    # Nutation: R1(-eps0) * R3(dpsi) * R1(eps)
    q = zxz2q(-eps0, dpsi, eps + deps, scalar=4)

    return q


def earth_nutation_precession_q(jd0: np.ndarray, jd1: np.ndarray, const, scalar: int = 4) -> np.ndarray:
    """
    # Earth rotation including precession and nutation (MATLAB: earthNutationPrecessionQ)

    Parameters
    ----------
    jd0 : np.ndarray
        Julian day, day (epoch)
    jd1 : np.ndarray
        Julian day, day (target)
    const : OrbitalConstants
        orbital constants
    scalar : int
        quaternion definition (0 or 4)

    Returns
    -------
    q : np.ndarray
        quaternion from jd0 to jd1, nx4
    """
    from ..attitude import q_mult

    assert scalar in [0, 4], "quaternion definition is unclear"

    q_n = nutation_q(jd1, const)
    q_p = precession_q(jd0, jd1, const)

    q = q_mult(q_n, q_p, scalar=4, definition=1)

    # Ensure 2D
    q = np.atleast_2d(q)

    if scalar == 0:
        q = np.column_stack([q[:, 3], q[:, 0:3]])

    return q


def earthNutationPrecessionQ(jd0: np.ndarray, jd1: np.ndarray, scalar: int, const) -> np.ndarray:
    """
    # MATLAB-compatible alias of earth_nutation_precession_q
    """
    return earth_nutation_precession_q(jd0, jd1, const, scalar=scalar)


def obliquity(jd: np.ndarray, const) -> np.ndarray:
    """
    # Mean obliquity of the ecliptic

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    eps0 : np.ndarray
        mean obliquity, rad

    References
    ----------
    Montenbruck, O., and Gill, E., Satellite Orbits

    Revisions
    ---------
    20211027  y.yoshimura
    """
    T = (jd - const.J2000) / 36525.0

    # Mean obliquity of the ecliptic, arcsec
    eps0_arcs = 84381.448 - 46.8150 * T - 0.00059 * T ** 2 + 0.001813 * T ** 3
    eps0 = arcs2rad(eps0_arcs)

    return eps0


# %[appendix]{"version":"1.0"}
