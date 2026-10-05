"""
Orbital disturbance accelerations versus distance from the Earth
Python conversion from yMATLAB/environment/disturbances.m
"""

import numpy as np

from ..conversion import au2km


def disturbances(const=None, h: np.ndarray = None, plot: bool = True) -> dict:
    """
    # Typical "magnitude of disturbance accelerations" figure (work in progress)

    The spacecraft, the Moon and the Sun are assumed to lie on the x-axis of
    the inertial frame. The disturbances are computed as vectors and their
    norms are plotted.

    Parameters
    ----------
    const : OrbitalConstants, optional
        orbital constants (default: orbit_const())
    h : np.ndarray, optional
        altitude, km (default: 0:10:43000)
    plot : bool, optional
        plot the norms of the accelerations (default: True)

    Returns
    -------
    out : dict
        - r: distance from the center of the Earth, km, n
        - a_earth: two-body acceleration, km/s^2, n x 3
        - a_j2: J2 acceleration, km/s^2, n x 3
        - a_sun: third-body acceleration by the Sun, km/s^2, n x 3
        - a_moon: third-body acceleration by the Moon, km/s^2, n x 3

    Notes
    -----
    MATLAB: disturbances.m is a script; here the computed accelerations are
    returned so that they can be used without the figure.

    Revisions
    ---------
    20260911  y.yoshimura, sign of dU/dphi of J2 fixed (C20 = -J2)

    See also
    --------
    orbit_const
    """
    if const is None:
        from ..orbit import orbit_const
        const = orbit_const()

    mu = const.GE
    J2 = const.J2

    if h is None:
        h = np.arange(0, 43000 + 10, 10)  # altitude
    h = np.atleast_1d(np.asarray(h, dtype=float)).flatten()

    r = (const.RE + h)[:, np.newaxis]  # distance from the center of the Earth
    r_vec = np.hstack([np.ones_like(r), np.zeros((len(r), 2))]) * r
    x = r_vec[:, 0:1]
    y = r_vec[:, 1:2]
    z = r_vec[:, 2:3]

    # latitude
    phi = 0.0

    sun_pos = au2km(1, const) * np.array([1.0, 0.0, 0.0])
    moon_pos = 384400 * np.array([1.0, 0.0, 0.0])  # km

    # Earth
    a_earth = -const.GE / r ** 3 * r_vec

    # J2
    dUdr = J2 * mu / r ** 2 * 3 * (const.RE / r) ** 2 / 2 * (3 * np.sin(phi) ** 2 - 1)
    # minus sign because C20 = -J2
    dUdphi = -J2 * mu / r * (const.RE / r) ** 2 * 3 * np.sin(phi) * np.sqrt(1 - np.sin(phi) ** 2)
    dUdlam = 0.0

    xy = np.sqrt(x ** 2 + y ** 2)
    a_j2x = (1 / r * dUdr - z / r ** 2 / xy * dUdphi) * x - y / (x ** 2 + y ** 2) * dUdlam
    a_j2y = (1 / r * dUdr - z / r ** 2 / xy * dUdphi) * y + x / (x ** 2 + y ** 2) * dUdlam
    a_j2z = z / r * dUdr + xy / r ** 2 * dUdphi

    a_j2 = np.hstack([a_j2x, a_j2y, a_j2z])

    # Sun
    # relative position vector from satellite to sun at inertial frame
    r_sc2s = sun_pos - r_vec  # n x 3
    a_sun = const.GS * (r_sc2s / np.linalg.norm(r_sc2s, axis=1, keepdims=True) ** 3
                        - sun_pos / np.linalg.norm(sun_pos) ** 3)

    # Moon
    # relative position vector from satellite to moon at inertial frame
    r_sc2m = moon_pos - r_vec  # n x 3
    a_moon = const.GM * (r_sc2m / np.linalg.norm(r_sc2m, axis=1, keepdims=True) ** 3
                         - moon_pos / np.linalg.norm(moon_pos) ** 3)

    r = r.flatten()

    if plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.semilogy(r, np.linalg.norm(a_earth, axis=1), 'b', label='Earth')
        plt.semilogy(r, np.linalg.norm(a_j2, axis=1), 'b--', label='$J_2$')
        plt.semilogy(r, np.linalg.norm(a_sun, axis=1), 'r', label='Sun')
        plt.semilogy(r, np.linalg.norm(a_moon, axis=1), 'k', label='Moon')
        plt.legend()
        plt.xlabel('distance from the Earth')
        plt.ylabel('acceleration, km/s^2')

    return {'r': r, 'a_earth': a_earth, 'a_j2': a_j2, 'a_sun': a_sun, 'a_moon': a_moon}


# %[appendix]{"version":"1.0"}
