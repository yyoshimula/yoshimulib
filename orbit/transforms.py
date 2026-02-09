"""
Coordinate transformations
Python conversion from yMATLAB/orbit/itrf2gcrf.m, dcmI2RTN.m, shadow.m, etc.
"""

import numpy as np
from .constants import arcs2rad


def earth_w(lod: np.ndarray) -> np.ndarray:
    """
    # Earth angular rate using LOD (length of day)

    Parameters
    ----------
    lod : np.ndarray
        length of day, n x 1 vector, s

    Returns
    -------
    w : np.ndarray
        angular rate norm, n x 1 vector, rad/s

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics and
    Applications. Springer Science & Business Media. 4th edition, p.222

    Revisions
    ---------
    20230608  y.yoshimura, y.yoshimula@gmail.com
    """
    lod = np.atleast_1d(lod)
    w = 7.292115146706979e-5 * (1 - lod / 86400)
    return w


def wobble(t_tt: np.ndarray) -> np.ndarray:
    """
    # Annual wobble for polar motion of the Earth

    IAU-2006/2000, CIO-based

    Parameters
    ----------
    t_tt : np.ndarray
        Julian century

    Returns
    -------
    s_prime : np.ndarray
        rad

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics and
    Applications. Springer Science & Business Media. 4th edition, p.212

    Revisions
    ---------
    20230605  y.yoshimura, y.yoshimula@gmail.com
    """
    t_tt = np.atleast_1d(t_tt)
    s_prime = arcs2rad(-0.000047) * t_tt
    return s_prime


def shadow(sat_i: np.ndarray, sun_i: np.ndarray, r_s: float, r_e: float) -> np.ndarray:
    """
    # Earth shadow function

    Parameters
    ----------
    sat_i : np.ndarray
        satellite position @ inertial frame, arbitrary unit, n x 3
    sun_i : np.ndarray
        sun position @ inertial frame, n x 3
    r_s : float
        Sun radius, scalar
    r_e : float
        Earth radius, scalar

    Returns
    -------
    nu : np.ndarray
        shadow function: 1 = sunlit, 0 = umbra (eclipse), n x 1

    Notes
    -----
    All variables (sat_i, sun_i, r_s, r_e) must have the same unit.
    変数の単位は統一すること。（統一していればmでもkmでもok）

    References
    ----------
    Satellite Orbits, Montenbruck, Gill, p.81

    Revisions
    ---------
    20231128  y.yoshimura: 引数を変更
    20211027  y.yoshimura, y.yoshimula@gmail.com
    """
    sat_i = np.atleast_2d(sat_i)
    sun_i = np.atleast_2d(sun_i)

    sun_rel = sun_i - sat_i

    # Sun angular radius
    a = np.arcsin(r_s / np.linalg.norm(sun_rel, axis=1))
    # Earth angular radius
    b = np.arcsin(r_e / np.linalg.norm(sat_i, axis=1))
    # Separation
    sat_norm = np.linalg.norm(sat_i, axis=1)
    sun_rel_norm = np.linalg.norm(sun_rel, axis=1)
    c = np.arccos(np.clip(
        np.sum(-sat_i * sun_rel, axis=1) / (sat_norm * sun_rel_norm),
        -1, 1
    ))

    n = sat_i.shape[0]
    nu = np.ones(n)

    # Total occultation (Sun fully covered): c <= b - a
    idx_total = c <= (b - a)
    nu[idx_total] = 0

    # Annular-like (Earth fully inside Sun disk): c <= a - b
    idx_annular = (c <= (a - b)) & ~idx_total
    nu[idx_annular] = 1 - (b[idx_annular] ** 2 / a[idx_annular] ** 2)

    # Partial overlap: |a-b| < c < a+b
    idx_part = (np.abs(a - b) < c) & (c < (a + b))
    if np.any(idx_part):
        ai = a[idx_part]
        bi = b[idx_part]
        ci = c[idx_part]

        u = (ci ** 2 + ai ** 2 - bi ** 2) / (2 * ci)
        v = (ci ** 2 + bi ** 2 - ai ** 2) / (2 * ci)

        # Clip for numerical safety
        t1 = np.clip(u / ai, -1, 1)
        t2 = np.clip(v / bi, -1, 1)

        w = np.maximum(0, ai ** 2 - u ** 2)

        A = ai ** 2 * np.arccos(t1) + bi ** 2 * np.arccos(t2) - ci * np.sqrt(w)
        nu[idx_part] = 1 - A / (np.pi * ai ** 2)

    # No overlap already nu=1: c >= a+b

    return nu


def dcm_i2rtn(raan: np.ndarray, inc: np.ndarray, w: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """
    # DCM from inertial frame to RTN frame

    Parameters
    ----------
    raan : np.ndarray
        right ascension of ascending node, rad
    inc : np.ndarray
        inclination, rad
    w : np.ndarray
        argument of perigee, rad
    nu : np.ndarray
        true anomaly, rad

    Returns
    -------
    R : np.ndarray
        DCM from inertial to RTN, 3x3

    Revisions
    ---------
    20211027  y.yoshimura
    """
    from ..attitude import zxz2dcm

    R = zxz2dcm(raan, inc, w + nu)
    return R


def ecef2lat_lon_h(r_ecef: np.ndarray, const) -> tuple:
    """
    # Convert ECEF position to geodetic latitude, longitude, and altitude

    Parameters
    ----------
    r_ecef : np.ndarray
        position in ECEF frame, km, n x 3
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    lat : np.ndarray
        geodetic latitude, rad
    lon : np.ndarray
        geodetic longitude, rad
    h : np.ndarray
        altitude above ellipsoid, km

    References
    ----------
    Vallado, D.A., "Fundamentals of Astrodynamics and Applications"

    Revisions
    ---------
    20211027  y.yoshimura
    """
    r_ecef = np.atleast_2d(r_ecef)

    x = r_ecef[:, 0]
    y = r_ecef[:, 1]
    z = r_ecef[:, 2]

    # Earth parameters
    a = const.RE  # equatorial radius, km
    f = const.fE  # flattening
    e2 = f * (2 - f)  # eccentricity squared

    # Longitude
    lon = np.arctan2(y, x)

    # Iterative algorithm for latitude and altitude
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e2))  # initial estimate

    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))

    # Final altitude calculation
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N

    return lat, lon, h


def geocentric2geodetic(lat_gc: np.ndarray, const) -> np.ndarray:
    """
    # Convert geocentric latitude to geodetic latitude

    Parameters
    ----------
    lat_gc : np.ndarray
        geocentric latitude, rad
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    lat_gd : np.ndarray
        geodetic latitude, rad

    Revisions
    ---------
    20211027  y.yoshimura
    """
    f = const.fE
    lat_gd = np.arctan(np.tan(lat_gc) / (1 - f) ** 2)
    return lat_gd


def geodetic2geocentric(lat_gd: np.ndarray, const) -> np.ndarray:
    """
    # Convert geodetic latitude to geocentric latitude

    Parameters
    ----------
    lat_gd : np.ndarray
        geodetic latitude, rad
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    lat_gc : np.ndarray
        geocentric latitude, rad

    Revisions
    ---------
    20211027  y.yoshimura
    """
    f = const.fE
    lat_gc = np.arctan((1 - f) ** 2 * np.tan(lat_gd))
    return lat_gc


def earth_g(jd: float, r_vec: np.ndarray, const,
             egm: dict) -> np.ndarray:
    """
    # Earth gravitational force w.r.t. ECEF frame

    Parameters
    ----------
    jd : float
        Julian day, day
    r_vec : np.ndarray
        satellite position at inertial frame, km, n x 3 vector
    const : OrbitalConstants
        orbital constants
    egm : dict
        earth gravity constants with keys:
        - 'geo_deg': degree of geopotential
        - 'Cnm': gravitational coefficients C
        - 'Snm': gravitational coefficients S

    Returns
    -------
    a_earth : np.ndarray
        Earth's gravitational force w.r.t. ECEF frame, km/s^2, n x 3 vector

    Notes
    -----
    Calculates non-spherical Earth gravitational acceleration using EGM model.

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    orbit_const, egm2008, precession_dcm, nutation_dcm, gast
    """
    from .precession import nutation_dcm, precession_dcm
    from .sidereal import gast
    from .gravity import egm2008
    from ..attitude import dcm_1axis

    r_vec = np.atleast_2d(r_vec)

    # Transform from inertial to PEF (pseudo-Earth-fixed)
    dcm_nutation = nutation_dcm(jd, const)  # nutation
    dcm_precession = precession_dcm(const.J2000, jd, const)  # precession
    GAST = gast(jd, const)
    i2pef = dcm_1axis(3, GAST) @ dcm_nutation @ dcm_precession

    r_pef = i2pef @ r_vec.T  # 3 x n

    # Calculate EGM2008 acceleration
    a_earth = egm2008(r_pef.T.flatten(), egm['geo_deg'], egm['Cnm'], egm['Snm'], const)

    return a_earth


# %[appendix]{"version":"1.0"}
