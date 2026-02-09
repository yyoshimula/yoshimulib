"""
Initial Orbit Determination methods
Python conversion from yMATLAB/orbitDetermination/
"""

import numpy as np


def gibbs(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, mu: float
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Gibbs method of orbit determination from three position vectors

    Parameters
    ----------
    r1 : np.ndarray
        geocentric position vector at t1, 1 x 3
    r2 : np.ndarray
        geocentric position vector at t2, 1 x 3
    r3 : np.ndarray
        geocentric position vector at t3, 1 x 3
    mu : float
        gravitational constant (unit must be consistent with position vectors)

    Returns
    -------
    v1 : np.ndarray
        velocity at t1, 1 x 3
    v2 : np.ndarray
        velocity at t2, 1 x 3
    v3 : np.ndarray
        velocity at t3, 1 x 3

    Notes
    -----
    Position vectors must be in chronological order: t1 < t2 < t3

    References
    ----------
    Curtis, Howard D. Orbital Mechanics for Engineering Students.
    Butterworth-Heinemann, 2013, p231-237.

    Revisions
    ---------
    20221110  y.yoshimura

    See also
    --------
    gauss, double_r
    """
    r1 = np.asarray(r1).flatten()
    r2 = np.asarray(r2).flatten()
    r3 = np.asarray(r3).flatten()

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    r3_norm = np.linalg.norm(r3)

    u1 = r1 / r1_norm

    c23 = np.cross(r2, r3)

    # Check coplanarity
    if np.abs(np.dot(u1, c23)) > 1e-3:
        raise ValueError("Position vectors are not coplanar.")

    # Calculate N, D, and S vectors
    N = r1_norm * np.cross(r2, r3) + r2_norm * np.cross(r3, r1) + r3_norm * np.cross(r1, r2)
    D = np.cross(r1, r2) + np.cross(r2, r3) + np.cross(r3, r1)
    S = r1 * (r2_norm - r3_norm) + r2 * (r3_norm - r1_norm) + r3 * (r1_norm - r2_norm)

    coef = np.sqrt(mu / np.linalg.norm(N) / np.linalg.norm(D))

    v1 = coef * (np.cross(D, r1) / r1_norm + S)
    v2 = coef * (np.cross(D, r2) / r2_norm + S)
    v3 = coef * (np.cross(D, r3) / r3_norm + S)

    return v1, v2, v3


def gauss(t: np.ndarray, azi_ele: np.ndarray, obs_eci: np.ndarray,
          r_init: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    # Gauss method of orbit determination

    Parameters
    ----------
    t : np.ndarray
        time at observations, 3 x 1 vector
    azi_ele : np.ndarray
        azimuth and elevation angles of the object at topocentric equatorial
        frame, 3 x 2 matrix
    obs_eci : np.ndarray
        observer position vector at inertial frame, 3 x 3 matrix
    r_init : float
        initial guess for r2 magnitude
    mu : float
        gravitational constant

    Returns
    -------
    r2 : np.ndarray
        position vector at t2, 1 x 3
    v2 : np.ndarray
        velocity vector at t2, 1 x 3

    Notes
    -----
    Uses Newton-Raphson iteration to solve 8th order polynomial.

    References
    ----------
    Curtis, Howard D. Orbital Mechanics for Engineering Students.
    Butterworth-Heinemann, 2013, p268.

    Revisions
    ---------
    20221110  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gibbs
    """
    TOL = 1e-8  # tolerance for 8th order equation

    # Time intervals
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]

    tau1 = t1 - t2
    tau3 = t3 - t2
    tau = tau3 - tau1

    # Unit vectors from spherical coordinates
    azi = azi_ele[:, 0]
    ele = azi_ele[:, 1]
    rho = np.column_stack([
        np.cos(azi) * np.cos(ele),
        np.sin(azi) * np.cos(ele),
        np.sin(ele)
    ])

    rho1 = rho[0, :]
    rho2 = rho[1, :]
    rho3 = rho[2, :]

    p1 = np.cross(rho2, rho3)
    p2 = np.cross(rho1, rho3)
    p3 = np.cross(rho1, rho2)

    # Calculate D
    R1 = obs_eci[0, :]
    R2 = obs_eci[1, :]
    R3 = obs_eci[2, :]

    D0 = np.dot(rho1, p1)

    D11 = np.dot(R1, p1)
    D12 = np.dot(R1, p2)
    D13 = np.dot(R1, p3)

    D21 = np.dot(R2, p1)
    D22 = np.dot(R2, p2)
    D23 = np.dot(R2, p3)

    D31 = np.dot(R3, p1)
    D32 = np.dot(R3, p2)
    D33 = np.dot(R3, p3)

    # Calculate A and B (in Eq.(5.112a))
    A = (-D12 * tau3 / tau + D22 + D32 * tau1 / tau) / D0
    B = (D12 * (tau3 ** 2 - tau ** 2) * tau3 / tau + D32 * (tau ** 2 - tau1 ** 2) * tau1 / tau) / 6 / D0

    # Calculate E, a, b, and c
    E = np.dot(R2, rho2)

    a_coef = -(A ** 2 + 2 * A * E + np.linalg.norm(R2) ** 2)
    b_coef = -2 * mu * B * (A + E)
    c_coef = -mu ** 2 * B ** 2

    # Solve 8th order equation using Newton-Raphson
    x = r_init
    residual = 1

    while np.abs(residual) > TOL:
        fx = x ** 8 + a_coef * x ** 6 + b_coef * x ** 3 + c_coef
        dfx = 8 * x ** 7 + 6 * a_coef * x ** 5 + 3 * b_coef * x ** 2

        residual = fx / dfx
        x = x - residual

    r2_norm = x

    # Slant ranges
    rho1_norm = (6 * (D31 * tau1 / tau3 + D21 * tau / tau3) * r2_norm ** 3 +
                 mu * D31 * (tau ** 2 - tau1 ** 2) * tau1 / tau3) / \
                (6 * r2_norm ** 3 + mu * (tau ** 2 - tau3 ** 2)) - D11
    rho1_norm = rho1_norm / D0

    rho2_norm = A + mu * B / r2_norm ** 3

    rho3_norm = (6 * (D13 * tau3 / tau1 - D23 * tau / tau1) * r2_norm ** 3 +
                 mu * D13 * (tau ** 2 - tau3 ** 2) * tau3 / tau1) / \
                (6 * r2_norm ** 3 + mu * (tau ** 2 - tau1 ** 2)) - D33
    rho3_norm = rho3_norm / D0

    # Position vectors at inertial frame
    r1 = R1 + rho1_norm * rho1
    r2 = R2 + rho2_norm * rho2
    r3 = R3 + rho3_norm * rho3

    # Lagrange coefficients
    f1 = 1 - 0.5 * mu / r2_norm ** 3 * tau1 ** 2
    f3 = 1 - 0.5 * mu / r2_norm ** 3 * tau3 ** 2

    g1 = tau1 - mu / 6 / r2_norm ** 3 * tau1 ** 3
    g3 = tau3 - mu / 6 / r2_norm ** 3 * tau3 ** 3

    # Velocity vectors
    v2 = 1 / (f1 * g3 - f3 * g1) * (-f3 * r1 + f1 * r3)

    return r2, v2


def double_r(t: np.ndarray, azi_ele: np.ndarray, r_obs: np.ndarray,
             mu: float, r_earth: float) -> tuple[np.ndarray, np.ndarray]:
    """
    # Double-r method of orbit determination

    Parameters
    ----------
    t : np.ndarray
        time at observations, 3 x 1 vector (Julian day)
    azi_ele : np.ndarray
        azimuth and elevation angles of the object at topocentric equatorial
        frame, 3 x 2 matrix
    r_obs : np.ndarray
        observer position vector at inertial frame, 3 x 3 matrix
    mu : float
        gravitational constant
    r_earth : float
        Earth radius (for convergence tolerance)

    Returns
    -------
    r2 : np.ndarray
        position vector at t2, 1 x 3
    v2 : np.ndarray
        velocity vector at t2, 1 x 3

    Notes
    -----
    Uses iterative Newton-Raphson method.

    References
    ----------
    Vallado, D.A., & Wayne D. McClain. Fundamentals of Astrodynamics and
    Applications. Springer Science & Business Media, 2001, pp443-445.

    Revisions
    ---------
    20221202  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gibbs, gauss
    """
    TOL = 1e-8 * r_earth

    # Time intervals (convert from days to seconds)
    t1 = t[0]
    t2 = t[1]
    t3 = t[2]

    tau1 = (t1 - t2) * 86400  # s
    tau3 = (t3 - t2) * 86400

    # Initial guess
    r = np.array([2.0 * r_earth, 2.01 * r_earth])

    # Unit vectors
    azi = azi_ele[:, 0]
    ele = azi_ele[:, 1]
    L = np.column_stack([
        np.cos(azi) * np.cos(ele),
        np.sin(azi) * np.cos(ele),
        np.sin(ele)
    ])

    # Parameters
    c = 2 * np.sum(L[:2] * r_obs[:2], axis=1)

    r_old = np.array([9999.99, 9999.99])

    # Iteration
    max_iter = 100
    for _ in range(max_iter):
        if np.abs(r[0] - r_old[0]) <= TOL and np.abs(r[1] - r_old[1]) <= TOL:
            break

        F1, F2, r_vec, a, dlt_e = _calc_f_double_r(tau1, tau3, c, r_obs, r, L, mu)

        # Partial derivative w.r.t. r1
        dlt_r1 = 0.005 * r[0]
        dF1_r1, dF2_r1, _, _, _ = _calc_f_double_r(tau1, tau3, c, r_obs, np.array([r[0] + dlt_r1, r[1]]), L, mu)

        dF1dr1 = (dF1_r1 - F1) / dlt_r1
        dF2dr1 = (dF2_r1 - F2) / dlt_r1

        # Partial derivative w.r.t. r2
        dlt_r2 = 0.005 * r[1]
        dF1_r2, dF2_r2, _, _, _ = _calc_f_double_r(tau1, tau3, c, r_obs, np.array([r[0], r[1] + dlt_r2]), L, mu)

        dF1dr2 = (dF1_r2 - F1) / dlt_r2
        dF2dr2 = (dF2_r2 - F2) / dlt_r2

        # Evaluate
        dlt = dF1dr1 * dF2dr2 - dF2dr1 * dF1dr2

        dlt1 = dF2dr2 * F1 - dF1dr2 * F2
        dlt2 = dF1dr1 * F2 - dF2dr1 * F1

        r_old = r.copy()

        # Update
        r[0] = r[0] + dlt1
        r[1] = r[1] + dlt2

    # Final outputs: velocity vector
    _, _, r_vec, a, dlt_e = _calc_f_double_r(tau1, tau3, c, r_obs, r, L, mu)

    f = 1 - a / r[1] * (1 - np.cos(dlt_e))
    g = tau3 - np.sqrt(a ** 3 / mu) * (dlt_e - np.sin(dlt_e))

    r2 = r_vec[1, :]
    v2 = (r_vec[2, :] - f * r_vec[1, :]) / g

    return r2, v2


def _calc_f_double_r(tau1, tau3, c, r_obs, r, L, mu):
    """Helper function for double_r method."""
    r = np.atleast_1d(r)
    c = np.atleast_1d(c)

    # Calculate rho
    rho_obs_norm = np.linalg.norm(r_obs[:2], axis=1)
    rho = (-c[:2] + np.sqrt(c[:2] ** 2 - 4 * (rho_obs_norm ** 2 - r[:2] ** 2))) / 2

    r_vec = np.zeros((3, 3))
    r_vec[:2] = rho[:, np.newaxis] * L[:2] + r_obs[:2]

    W = np.cross(r_vec[0], r_vec[1])
    W = W / np.linalg.norm(r_vec[0]) / np.linalg.norm(r_vec[1])

    rho_3 = -np.dot(r_obs[2], W) / np.dot(L[2], W)
    r_vec[2] = rho_3 * L[2] + r_obs[2]
    r_3 = np.linalg.norm(r_vec[2])

    # Calculate angles
    cos_dlt_v = np.zeros((3, 2))
    sin_dlt_v = np.zeros((3, 2))
    for j in [1, 2]:
        for k in [0, 1]:
            r_j_norm = np.linalg.norm(r_vec[j])
            r_k_norm = np.linalg.norm(r_vec[k])
            cos_dlt_v[j, k] = np.dot(r_vec[j], r_vec[k]) / r_j_norm / r_k_norm
            sin_dlt_v[j, k] = np.sqrt(1 - cos_dlt_v[j, k] ** 2)

    dlt_v_21 = np.arctan2(sin_dlt_v[1, 0], cos_dlt_v[1, 0])
    dlt_v_31 = np.arctan2(sin_dlt_v[2, 0], cos_dlt_v[2, 0])

    r_norms = np.array([np.linalg.norm(r_vec[0]), np.linalg.norm(r_vec[1]), r_3])

    if dlt_v_31 > np.pi:
        c1 = r_norms[1] * sin_dlt_v[2, 1] / r_norms[0] / sin_dlt_v[2, 0]
        c3 = r_norms[1] * sin_dlt_v[1, 0] / r_norms[2] / sin_dlt_v[2, 0]
        p = (c1 * r_norms[0] + c3 * r_norms[2] - r_norms[1]) / (c1 + c3 - 1)
    else:
        c1 = r_norms[0] * sin_dlt_v[2, 0] / r_norms[1] / sin_dlt_v[2, 1]
        c3 = r_norms[0] * sin_dlt_v[1, 0] / r_norms[2] / sin_dlt_v[2, 1]
        p = (c3 * r_norms[2] - c1 * r_norms[1] + r_norms[0]) / (-c1 + c3 + 1)

    ecosV = p / r_norms - 1

    if np.deg2rad(179) <= dlt_v_21 <= np.deg2rad(181):
        esinV_2 = (cos_dlt_v[2, 1] * ecosV[1] - ecosV[2]) / sin_dlt_v[2, 1]
    else:
        esinV_2 = (-cos_dlt_v[1, 0] * ecosV[1] + ecosV[0]) / sin_dlt_v[1, 0]

    e = np.sqrt(ecosV[1] ** 2 + esinV_2 ** 2)
    a = p / (1 - e ** 2)
    n = np.sqrt(mu / a ** 3)

    S = r_norms[1] / p * np.sqrt(1 - e ** 2) * esinV_2
    C = r_norms[1] / p * (e ** 2 + ecosV[1])

    sin_dlt_E_32 = r_norms[2] / np.sqrt(a * p) * sin_dlt_v[2, 1] - r_norms[2] / p * (1 - cos_dlt_v[2, 1]) * S
    cos_dlt_E_32 = 1 - r_norms[1] * r_norms[2] / (a * p) * (1 - cos_dlt_v[2, 1])

    sin_dlt_E_21 = r_norms[0] / np.sqrt(a * p) * sin_dlt_v[1, 0] + r_norms[0] / p * (1 - cos_dlt_v[1, 0]) * S
    cos_dlt_E_21 = 1 - r_norms[1] * r_norms[0] / (a * p) * (1 - cos_dlt_v[1, 0])

    dlt_E_32 = np.arctan2(sin_dlt_E_32, cos_dlt_E_32)
    dlt_E_21 = np.arctan2(sin_dlt_E_21, cos_dlt_E_21)

    dlt_M_32 = dlt_E_32 + 2 * S * np.sin(dlt_E_32 / 2) ** 2 - C * np.sin(dlt_E_32)
    dlt_M_12 = -dlt_E_21 + 2 * S * np.sin(dlt_E_21 / 2) ** 2 + C * np.sin(dlt_E_21)

    F1 = tau1 - dlt_M_12 / n
    F2 = tau3 - dlt_M_32 / n

    return F1, F2, r_vec, a, dlt_E_32


def calcF(tau1, tau3, c, r_obs, r, L, mu):
    """
    # MATLAB-compatible wrapper for Double-r helper (MATLAB: calcF)
    """
    return _calc_f_double_r(tau1, tau3, c, r_obs, r, L, mu)


# %[appendix]{"version":"1.0"}
