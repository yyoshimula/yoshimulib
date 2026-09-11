"""
Orbital elements conversions (OE <-> Position/Velocity)
Python conversion from yMATLAB/orbit/
"""

import numpy as np
from .kepler import true_anomaly, mean_anomaly
from ..attitude import zxz2q, q_inv, q_rotation


def oe2rv(oe: np.ndarray, flag: int, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    # Converting absolute orbital elements to position and velocity vector

    Parameters
    ----------
    oe : np.ndarray
        orbital elements [a, e, i, RAAN, w, f (or M)], n x 6
        a: semi-major axis (m or km)
        e: eccentricity
        i: inclination (rad)
        RAAN: right ascension of ascending node (rad)
        w: argument of perigee (rad)
        f: true anomaly or M: mean anomaly (rad)
    flag : int
        1 := true anomaly, 0 := mean anomaly
    mu : float
        gravitational constant (m^3/s^2 or km^3/s^2)

    Returns
    -------
    r : np.ndarray
        position vector, n x 3 (m or km)
    v : np.ndarray
        velocity vector, n x 3 (m/s or km/s)

    Notes
    -----
    Units must be consistent between mu and position/velocity.
    All supplied orientation angles are applied, including for circular and
    equatorial orbits. Although the individual angles are then non-unique,
    their combined orientation/phase must be preserved in the returned state.

    References
    ----------
    Vallado, D. A., and McClain, W. D., Fundamentals of Astrodynamics and Applications

    Revisions
    ---------
    20231204  y.yoshimura
    20211027  y.yoshimura

    See also
    --------
    rv2oe, true_anomaly
    """
    assert flag in [0, 1], "flag must be 0 or 1"

    oe = np.atleast_2d(oe)
    small = 1.0e-10

    a = oe[:, 0]
    e = oe[:, 1]
    inc = oe[:, 2]
    raan = oe[:, 3]
    ome = oe[:, 4]

    if flag == 1:  # true anomaly
        f = oe[:, 5]
    else:  # mean anomaly
        f, _ = true_anomaly(a, e, np.mod(oe[:, 5], 2 * np.pi))

    # Preserve the supplied orientation even for circular/equatorial orbits.
    # Individual angles are not unique there, but their combined phase still
    # determines the state. Zeroing omega or RAAN alone changes that state.

    # Semi-latus rectum
    p = a * (1 - e ** 2)
    p = np.where(np.abs(p) < small, 0.0001, p)

    temp = p / (1.0 + e * np.cos(f))

    # Position in orbital plane (PQW)
    r_pqw = np.column_stack([temp * np.cos(f), temp * np.sin(f), np.zeros(len(f))])
    v_pqw = np.column_stack([-np.sin(f) * np.sqrt(mu / p),
                              (e + np.cos(f)) * np.sqrt(mu / p),
                              np.zeros(len(f))])

    # Transformation to IJK by quaternion
    q = zxz2q(raan, inc, ome, scalar=4)  # quaternion from IJK to PQW
    q = q_inv(q, scalar=4)  # quaternion from PQW to IJK

    r = q_rotation(r_pqw, q, scalar=4)  # position @ inertial frame
    v = q_rotation(v_pqw, q, scalar=4)  # velocity @ inertial frame

    return r, v


# %[appendix]{"version":"1.0"}


def rv2oe(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """
    # position r and velocity v to orbital elements

    Parameters
    ----------
    r : np.ndarray
        position vector, n x 3 (m or km)
    v : np.ndarray
        velocity vector, n x 3 (m/s or km/s)
    mu : float
        gravitational constant (m^3/s^2 or km^3/s^2)

    Returns
    -------
    oe : np.ndarray
        orbital elements [a, e, inc, raan, w, nu], n x 6

    Notes
    -----
    NA

    References
    ----------
    Vallado, D.A., & Wayne D. McClain. Fundamentals of Astrodynamics and Applications.
    4th edition, Springer Science & Business Media, 2001. pp114.

    Revisions
    ---------
    20221110  y.yoshimura

    See also
    --------
    oe2rv
    """
    r = np.atleast_2d(r)
    v = np.atleast_2d(v)
    nt = r.shape[0]

    eps = np.finfo(float).eps

    # Orbital angular momentum
    h = np.cross(r, v)

    # Node vector
    k_vec = np.zeros((nt, 3))
    k_vec[:, 2] = 1.0
    n_vec = np.cross(k_vec, h)

    # Eccentricity vector
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    e_vec = ((v_norm ** 2 - mu / r_norm) * r - np.sum(r * v, axis=1, keepdims=True) * v) / mu

    # Eccentricity
    e = np.linalg.norm(e_vec, axis=1)

    # Specific mechanical energy
    xi = v_norm.flatten() ** 2 / 2 - mu / r_norm.flatten()

    # Semi-major axis
    a = np.where(np.abs(xi) > eps, -mu / (2 * xi), np.inf)

    # Semi-latus rectum
    h_norm = np.linalg.norm(h, axis=1)
    p = h_norm ** 2 / mu

    # Inclination
    inc = np.arccos(h[:, 2] / h_norm)

    # RAAN
    n_norm = np.linalg.norm(n_vec, axis=1)
    raan = np.arccos(np.clip(n_vec[:, 0] / np.maximum(n_norm, eps), -1, 1))
    raan = np.where(n_vec[:, 1] < 0, 2 * np.pi - raan, raan)

    # True anomaly
    rdot_e = np.sum(r * e_vec, axis=1) / (r_norm.flatten() * np.maximum(e, eps))
    nu = np.arccos(np.clip(rdot_e, -1, 1))
    rdot_v = np.sum(r * v, axis=1)
    nu = np.where(rdot_v < 0, 2 * np.pi - nu, nu)

    # Argument of perigee
    ndot_e = np.sum(n_vec * e_vec, axis=1) / (np.maximum(n_norm, eps) * np.maximum(e, eps))
    w = np.arccos(np.clip(ndot_e, -1, 1))
    w = np.where(e_vec[:, 2] < 0, 2 * np.pi - w, w)

    oe = np.column_stack([a, e, inc, raan, w, nu])

    return oe


# %[appendix]{"version":"1.0"}


# Gravitational constants
MU_EARTH_KM = 398600.4418  # km^3/s^2
MU_EARTH_M = 3.986004418e14  # m^3/s^2
MU_SUN_KM = 1.32712440018e11  # km^3/s^2
MU_MOON_KM = 4902.800066  # km^3/s^2

# Earth parameters
R_EARTH_KM = 6378.137  # km (equatorial radius)
R_EARTH_M = 6378137.0  # m (equatorial radius)
J2_EARTH = 1.08262668e-3  # J2 coefficient

# Astronomical constants
AU_KM = 149597870.7  # km
