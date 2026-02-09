"""
Kepler's equation and anomaly conversions
Python conversion from yMATLAB/orbit/
"""

import numpy as np


def kepler_eq(M: float | np.ndarray, e: float | np.ndarray, tol: float = 1e-8) -> float | np.ndarray:
    """
    # solving Kepler's equations

    Calculate eccentric anomaly from mean anomaly and eccentricity

    Parameters
    ----------
    M : float or np.ndarray
        mean anomaly, rad
    e : float or np.ndarray
        eccentricity
    tol : float, optional
        tolerance (default: 1e-8)

    Returns
    -------
    E : float or np.ndarray
        eccentric anomaly, rad

    Notes
    -----
    Kepler's equation: M = E - e*sin(E)

    References
    ----------
    Curtis, Howard D 2013 Orbital mechanics for engineering students, p148 Algorithm 3.1

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    true_anomaly
    """
    M = np.asarray(M)
    e = np.asarray(e)
    scalar_input = M.ndim == 0
    M = np.atleast_1d(M)
    e = np.atleast_1d(e)

    # Initial estimate based on M
    E = np.where(M < np.pi, M + 0.5 * e, M - 0.5 * e)

    # Newton-Raphson iteration
    residual = np.ones_like(E)
    max_iter = 100
    iteration = 0

    while np.any(np.abs(residual) > tol) and iteration < max_iter:
        fE = E - e * np.sin(E) - M
        dfE = 1 - e * np.cos(E)
        residual = fE / dfE
        E = E - residual
        iteration += 1

    # Wrap to [0, 2*pi]
    E = np.mod(E, 2 * np.pi)

    if scalar_input:
        return float(E[0])
    return E


# %[appendix]{"version":"1.0"}


def e_anomaly(e: float | np.ndarray, f: float | np.ndarray) -> float | np.ndarray:
    """
    # eccentric anomaly using eccentricity and true anomaly

    Parameters
    ----------
    e : float or np.ndarray
        eccentricity
    f : float or np.ndarray
        true anomaly, rad

    Returns
    -------
    E : float or np.ndarray
        eccentric anomaly, rad

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20221110  y.yoshimura

    See also
    --------
    kepler_eq
    """
    f = np.mod(f, 2 * np.pi)

    # Eccentric anomaly
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f / 2))

    E = np.mod(E, 2 * np.pi)

    return E


# %[appendix]{"version":"1.0"}


def true_anomaly(a: float | np.ndarray, e: float | np.ndarray,
                 M: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    # true anomaly from orbital elements

    Parameters
    ----------
    a : float or np.ndarray
        semi-major axis
    e : float or np.ndarray
        eccentricity
    M : float or np.ndarray
        mean anomaly, rad

    Returns
    -------
    f : float or np.ndarray
        true anomaly, rad
    E : float or np.ndarray
        eccentric anomaly, rad

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    kepler_eq, e_anomaly
    """
    # Solve Kepler's equation
    E = kepler_eq(M, e)

    # True anomaly from eccentric anomaly
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))

    f = np.mod(f, 2 * np.pi)

    return f, E


# %[appendix]{"version":"1.0"}


def mean_anomaly(e: float | np.ndarray, f: float | np.ndarray) -> float | np.ndarray:
    """
    # mean anomaly from eccentricity and true anomaly

    Parameters
    ----------
    e : float or np.ndarray
        eccentricity
    f : float or np.ndarray
        true anomaly, rad

    Returns
    -------
    M : float or np.ndarray
        mean anomaly, rad

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    e_anomaly, true_anomaly
    """
    # Eccentric anomaly
    E = e_anomaly(e, f)

    # Mean anomaly (Kepler's equation)
    M = E - e * np.sin(E)

    M = np.mod(M, 2 * np.pi)

    return M


# %[appendix]{"version":"1.0"}
