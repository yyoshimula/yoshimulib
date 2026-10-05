"""
High-fidelity SRP approximation using Cook-Torrance model
Python conversion from yMATLAB/hifiSRP/
"""

import numpy as np
from typing import Tuple, Any
from ..object import SatelliteModel


def _colon(start: float, stop: float, n: int) -> np.ndarray:
    """
    Integration bounds start:(stop - start)/n:stop of MATLAB, n + 1 points
    (np.arange with a float step may add a point beyond stop).
    """
    return np.linspace(start, stop, n + 1)


def ct_m(sat: SatelliteModel, v: np.ndarray, sun_b: np.ndarray) -> float:
    """
    # Calculating remaining term M in the Cook-Torrance model

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    v : np.ndarray
        reference vector, 1x3 vector
    sun_b : np.ndarray
        sun vector from satellite to Sun expressed with body-fixed frame, 1x3 vector

    Returns
    -------
    M : float
        remaining term M(v) = G(v)F(v) / 4

    Notes
    -----
    Cook-Torrance model's non-NDF term calculation

    References
    ----------
    Analytic Approximation of High-Fidelity Solar Radiation Pressure.

    Revisions
    ---------
    20200915  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_approx_ct
    """
    n = sat.normal  # normal vectors, nx3 matrix

    v = np.asarray(v).flatten()  # 3x1 vector
    sun_b = np.asarray(sun_b).flatten()  # 3x1 vector

    # Bisector vector, 3x1 vector
    h = v + sun_b
    h = h / np.linalg.norm(h)

    # Specular
    nest = (1 + np.sqrt(sat.F0)) / (1 - np.sqrt(sat.F0))
    g = np.sqrt(nest ** 2 + (np.dot(v, h)) ** 2 - 1)  # nx1

    temp1 = 2 * (n @ h) * (n @ v) / np.dot(v, h)
    temp2 = 2 * (n @ h) * (n @ sun_b) / np.dot(v, h)  # nx1

    G = np.minimum(1, temp1)
    G = np.minimum(G, temp2)  # nx1

    temp1 = (g - np.dot(v, h)) ** 2 / 2 / (g + np.dot(v, h)) ** 2
    temp2 = (1 + (np.dot(v, h) * (g + np.dot(v, h)) - 1) ** 2 / (np.dot(v, h) * (g - np.dot(v, h)) + 1) ** 2)
    F = temp1 * temp2

    M = np.sum(G * F) / 4

    return M


def ct_m2(sat: SatelliteModel, v: np.ndarray, sun_b: np.ndarray) -> np.ndarray:
    """
    # Calculating remaining term M in the Cook-Torrance model (vectorized)

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    v : np.ndarray
        reference vector, nFacet x 3 matrix
    sun_b : np.ndarray
        sun vector from satellite to Sun expressed with body-fixed frame, 1x3 vector

    Returns
    -------
    M : np.ndarray
        remaining term, nFacet x 1

    Notes
    -----
    Vectorized version of ct_m for multiple facets

    References
    ----------
    Analytic Approximation of High-Fidelity Solar Radiation Pressure.

    Revisions
    ---------
    20200915  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_approx_ct2
    """
    v = np.atleast_2d(v)
    sun_b = np.asarray(sun_b).flatten()

    # Bisector vector, nFacet x 3 matrix
    h = v + sun_b
    h = h / np.linalg.norm(h, axis=1, keepdims=True)

    # Pre calculation
    NS = sat.normal @ sun_b  # nFacet x 1
    NH = np.sum(sat.normal * h, axis=1)  # nFacet x 1
    NV = np.sum(sat.normal * v, axis=1)  # nFacet x 1
    VH = np.sum(v * h, axis=1)  # nFacet x 1

    # Specular
    nest = (1 + np.sqrt(sat.F0)) / (1 - np.sqrt(sat.F0))  # nFacet x 1
    g = np.sqrt(nest ** 2 + VH ** 2 - 1)  # nFacet x 1

    tmp1 = 2 * NH * NV / VH
    tmp2 = 2 * NH * NS / VH

    G = np.minimum(1, tmp1)
    G = np.minimum(G, tmp2)  # nFacet x 1

    tmp1 = (g - VH) ** 2 / 2 / (g + VH) ** 2
    tmp2 = (1 + (VH * (g + VH) - 1) ** 2 / (VH * (g - VH) + 1) ** 2)
    F = tmp1 * tmp2

    M = G * F / 4

    return M


def int_bound(phi: np.ndarray, theta: np.ndarray, n: np.ndarray,
              lobe_in: int) -> Tuple[float, float]:
    """
    # Calculate the integration bound

    Parameters
    ----------
    phi : np.ndarray
        phi range [phi0, phi1]
    theta : np.ndarray
        theta range [theta0, theta1]
    n : np.ndarray
        normal vector, 1x3
    lobe_in : int
        flag for lobe inclusion (1, 0, or -1)

    Returns
    -------
    a : float
        lower bound
    b : float
        upper bound

    Revisions
    ---------
    20201201  y.yoshimura
    """
    phi0, phi1 = phi[0], phi[1]
    theta0, theta1 = theta[0], theta[1]

    c0 = np.array([np.sin(theta0) * np.cos(phi0), np.sin(theta0) * np.sin(phi0), np.cos(theta0)])
    c1 = np.array([np.sin(theta0) * np.cos(phi1), np.sin(theta0) * np.sin(phi1), np.cos(theta0)])
    c2 = np.array([np.sin(theta1) * np.cos(phi1), np.sin(theta1) * np.sin(phi1), np.cos(theta1)])
    c3 = np.array([np.sin(theta1) * np.cos(phi0), np.sin(theta1) * np.sin(phi0), np.cos(theta1)])

    a = min(np.dot(c0, n), np.dot(c1, n), np.dot(c2, n), np.dot(c3, n))

    if lobe_in == 1:
        b = 1.0
    elif lobe_in == -1:
        a = -1.0
        b = max(np.dot(c0, n), np.dot(c1, n), np.dot(c2, n), np.dot(c3, n))
    else:
        b = max(np.dot(c0, n), np.dot(c1, n), np.dot(c2, n), np.dot(c3, n))

    return a, b


def calc_coeff(phi_bound: np.ndarray, theta_bound: np.ndarray, theta_n: float,
               n: np.ndarray, lam: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Calculate 1st-order approximation coefficients

    Parameters
    ----------
    phi_bound : np.ndarray
        phi boundary grid
    theta_bound : np.ndarray
        theta boundary grid
    theta_n : float
        angle between sun vector and normal vector
    n : np.ndarray
        normal vector, 1x3
    lam : float
        lambda coefficient
    mu : float
        mu coefficient

    Returns
    -------
    alp_ : np.ndarray
        alpha coefficients
    bet_ : np.ndarray
        beta coefficients

    Revisions
    ---------
    20210124  y.yoshimura
    """
    shape = (phi_bound.shape[0] - 1, phi_bound.shape[1] - 1)
    n_flag = np.zeros(shape)
    a = np.zeros(shape)
    b = np.zeros(shape)
    alp_ = np.zeros(shape)
    bet_ = np.zeros(shape)

    for i in range(shape[0]):
        for k in range(shape[1]):
            n_flag[i, k] = (phi_bound[i, k] <= np.pi / 2) * (np.pi / 2 <= phi_bound[i, k + 1])
            n_flag[i, k] = n_flag[i, k] * (theta_bound[i, k] <= theta_n) * (theta_n <= theta_bound[i + 1, k])
            a[i, k], b[i, k] = int_bound(
                [phi_bound[i, k], phi_bound[i, k + 1]],
                [theta_bound[i, k], theta_bound[i + 1, k]],
                n, int(n_flag[i, k])
            )
            p = (a[i, k] + b[i, k]) / 2
            q = (b[i, k] - a[i, k]) / 2

            if np.abs(lam * q) < 1e-10:
                r0 = mu * np.exp(lam * (p - 1))
                r1 = 0
            else:
                r0 = mu * np.exp(lam * (p - 1)) * np.sinh(lam * q) / (lam * q)
                r1 = 3 * mu * np.exp(lam * (p - 1)) * (lam * q * np.cosh(lam * q) - np.sinh(lam * q)) / (lam ** 2 * q ** 2)

            denom = b[i, k] - a[i, k]
            if np.abs(denom) < 1e-10:
                alp_[i, k] = r0
                bet_[i, k] = 0
            else:
                alp_[i, k] = r0 - r1 * (a[i, k] + b[i, k]) / denom  # alpha
                bet_[i, k] = 2 * r1 / denom  # beta

    return alp_, bet_


def analytic_sol_ct(theta_n: float, alp: float, bet: float,
                    phi: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Analytic solution for 1st order approximation

    Parameters
    ----------
    theta_n : float
        angle between sun vector and normal vector
    alp : float
        alpha coefficient
    bet : float
        beta coefficient
    phi : np.ndarray
        phi integration range [phi0, phi1]
    theta : np.ndarray
        theta integration range [theta0, theta1]

    Returns
    -------
    Axyz : np.ndarray
        A component, 1x3
    Bxyz : np.ndarray
        B component, 1x3

    Revisions
    ---------
    20210131  y.yoshimura
    """
    phi0, phi1 = phi[0], phi[1]
    theta0, theta1 = theta[0], theta[1]

    # A component
    Axyz = np.array([
        ((np.sin(phi0) - np.sin(phi1)) * (4 * theta0 - 4 * theta1 - np.sin(4 * theta0) + np.sin(4 * theta1))) / 4,
        -((np.cos(phi0) - np.cos(phi1)) * (4 * theta0 - 4 * theta1 - np.sin(4 * theta0) + np.sin(4 * theta1))) / 4,
        -((np.cos(4 * theta0) - np.cos(4 * theta1)) * (phi0 - phi1)) / 4
    ])
    Axyz = alp * Axyz

    # B component (complex analytic expression)
    c_t0_2 = np.cos(theta0 / 2)
    s_t0_2 = np.sin(theta0 / 2)
    c_t1_2 = np.cos(theta1 / 2)
    s_t1_2 = np.sin(theta1 / 2)
    c_tn_2 = np.cos(theta_n / 2)
    s_tn_2 = np.sin(theta_n / 2)
    s_tn = np.sin(theta_n)
    c_tn = np.cos(theta_n)
    c_t0 = np.cos(theta0)
    c_t1 = np.cos(theta1)
    s_t0 = np.sin(theta0)
    s_t1 = np.sin(theta1)

    Bx = ((64 * c_t0_2 ** 3 * s_t0_2 * np.sin(phi1)) / 3 - (64 * c_t0_2 ** 3 * s_t0_2 * np.sin(phi0)) / 3 +
          (64 * c_t1_2 ** 3 * s_t1_2 * np.sin(phi0)) / 3 + (1088 * c_t0_2 ** 5 * s_t0_2 * np.sin(phi0)) / 15 -
          (64 * c_t1_2 ** 3 * s_t1_2 * np.sin(phi1)) / 3 - (1088 * c_t0_2 ** 5 * s_t0_2 * np.sin(phi1)) / 15 -
          (1088 * c_t1_2 ** 5 * s_t1_2 * np.sin(phi0)) / 15 - (512 * c_t0_2 ** 7 * s_t0_2 * np.sin(phi0)) / 5 +
          (1088 * c_t1_2 ** 5 * s_t1_2 * np.sin(phi1)) / 15 + (512 * c_t0_2 ** 7 * s_t0_2 * np.sin(phi1)) / 5 +
          (512 * c_t1_2 ** 7 * s_t1_2 * np.sin(phi0)) / 5 + (256 * c_t0_2 ** 9 * s_t0_2 * np.sin(phi0)) / 5 -
          (512 * c_t1_2 ** 7 * s_t1_2 * np.sin(phi1)) / 5 - (256 * c_t0_2 ** 9 * s_t0_2 * np.sin(phi1)) / 5 -
          (256 * c_t1_2 ** 9 * s_t1_2 * np.sin(phi0)) / 5 + (256 * c_t1_2 ** 9 * s_t1_2 * np.sin(phi1)) / 5 +
          32 * c_t0_2 ** 4 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2 -
          32 * c_t0_2 ** 4 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2 -
          32 * c_t1_2 ** 4 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2 +
          32 * c_t1_2 ** 4 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2 -
          (320 * c_t0_2 ** 6 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2) / 3 +
          (320 * c_t0_2 ** 6 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2) / 3 +
          (320 * c_t1_2 ** 6 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2) / 3 -
          (320 * c_t1_2 ** 6 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2) / 3 +
          128 * c_t0_2 ** 8 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2 -
          128 * c_t0_2 ** 8 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2 -
          128 * c_t1_2 ** 8 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2 +
          128 * c_t1_2 ** 8 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2 -
          (256 * c_t0_2 ** 10 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2) / 5 +
          (256 * c_t0_2 ** 10 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2) / 5 +
          (256 * c_t1_2 ** 10 * c_tn_2 * s_tn_2 * np.cos(phi0) ** 2) / 5 -
          (256 * c_t1_2 ** 10 * c_tn_2 * s_tn_2 * np.cos(phi1) ** 2) / 5 +
          (128 * c_t0_2 ** 3 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi0)) / 3 -
          (128 * c_t0_2 ** 3 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi1)) / 3 -
          (128 * c_t1_2 ** 3 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi0)) / 3 -
          (2176 * c_t0_2 ** 5 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi0)) / 15 +
          (128 * c_t1_2 ** 3 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi1)) / 3 +
          (2176 * c_t0_2 ** 5 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi1)) / 15 +
          (2176 * c_t1_2 ** 5 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi0)) / 15 +
          (1024 * c_t0_2 ** 7 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi0)) / 5 -
          (2176 * c_t1_2 ** 5 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi1)) / 15 -
          (1024 * c_t0_2 ** 7 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi1)) / 5 -
          (1024 * c_t1_2 ** 7 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi0)) / 5 -
          (512 * c_t0_2 ** 9 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi0)) / 5 +
          (1024 * c_t1_2 ** 7 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi1)) / 5 +
          (512 * c_t0_2 ** 9 * c_tn_2 ** 2 * s_t0_2 * np.sin(phi1)) / 5 +
          (512 * c_t1_2 ** 9 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi0)) / 5 -
          (512 * c_t1_2 ** 9 * c_tn_2 ** 2 * s_t1_2 * np.sin(phi1)) / 5)

    term1 = (4 * c_t0 ** 3 * (3 * c_t0 ** 2 - 5)) / 15 - (4 * c_t1 ** 3 * (3 * c_t1 ** 2 - 5)) / 15
    term2 = (4 * s_t0 ** 3 * (3 * s_t0 ** 2 - 5)) / 15 - (4 * s_t1 ** 3 * (3 * s_t1 ** 2 - 5)) / 15

    By = ((np.sin(2 * phi1) * s_tn * term1) / 2 -
          (np.sin(2 * phi0) * s_tn * term1) / 2 +
          phi0 * s_tn * term1 - phi1 * s_tn * term1 +
          2 * np.cos(phi0) * c_tn * term2 - 2 * np.cos(phi1) * c_tn * term2)

    term3 = c_t0 ** 3 / 3 - c_t1 ** 3 / 3
    term4 = c_t0 ** 5 / 5 - c_t1 ** 5 / 5
    term5 = s_t0 ** 3 / 3 - s_t1 ** 3 / 3
    term6 = (s_t0 ** 3 * (3 * s_t0 ** 2 - 5)) / 15 - (s_t1 ** 3 * (3 * s_t1 ** 2 - 5)) / 15

    Bz = (4 * phi0 * c_tn * term3 - 4 * phi1 * c_tn * term3 -
          8 * phi0 * c_tn * term4 + 8 * phi1 * c_tn * term4 +
          4 * np.cos(phi0) * s_tn * term5 - 4 * np.cos(phi1) * s_tn * term5 +
          8 * np.cos(phi0) * s_tn * term6 - 8 * np.cos(phi1) * s_tn * term6)

    Bxyz = bet * np.array([Bx, By, Bz])

    return Axyz, Bxyz


def srp_approx_ct(sat: SatelliteModel, theta_n: float, sun_b: np.ndarray,
                  d: float, const: Any, n_theta: int = 6, n_phi: int = 6) -> np.ndarray:
    """
    # Approximating specular term of SRP with Cook-Torrance model

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    theta_n : float
        angle between sun vector and facet's normal vector
    sun_b : np.ndarray
        sun vector from satellite to Sun expressed with body-fixed frame, 1x3 vector
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        constants for orbital propagation
    n_theta : int, optional
        number of theta divisions (default: 6)
    n_phi : int, optional
        number of phi divisions (default: 6)

    Returns
    -------
    srp : np.ndarray
        approximated SRP with Cook-Torrance model expressed with Sun-fixed frame

    Notes
    -----
    Cook-Torrance model analytical SRP approximation

    References
    ----------
    Analytic Approximation of High-Fidelity Solar Radiation Pressure.

    Revisions
    ---------
    20200915  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_approx_ct2
    """
    from ..conversion import km2au

    # Coefficient
    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au ** 2

    sat_normal = np.array([0.0, np.sin(theta_n), np.cos(theta_n)])  # normal vector

    sun_b = np.asarray(sun_b).flatten()

    # Perfect mirror-like reflection vector
    r_ref = 2 * np.dot(sun_b, sat_normal) * sat_normal - sun_b

    # Create temporary satellite for ct_m
    class TempSat:
        pass

    temp_sat = TempSat()
    temp_sat.normal = sat_normal.reshape(1, 3)
    temp_sat.F0 = sat.F0[0] if hasattr(sat.F0, '__len__') else sat.F0

    M = ct_m(temp_sat, r_ref, sun_b)

    lam = 2 / sat.mCT[0] ** 2 if hasattr(sat.mCT, '__len__') else 2 / sat.mCT ** 2
    c_coef = 1

    sunlit_flag = float(np.dot(sat_normal, sun_b) > 0)

    # Quarter-sphere, +y direction integration range
    phi_vals = _colon(0, np.pi, n_phi)
    theta_vals = _colon(0, (np.pi / 2 + theta_n) / 2, n_theta)
    phi_bound, theta_bound = np.meshgrid(phi_vals, theta_vals)

    alp_, bet_ = calc_coeff(phi_bound, theta_bound, theta_n, sat_normal, lam, c_coef)

    A = np.zeros(3)
    B = np.zeros(3)
    for j in range(phi_bound.shape[0] - 1):
        for k in range(phi_bound.shape[1] - 1):
            Atmp, Btmp = analytic_sol_ct(
                theta_n, alp_[j, k], bet_[j, k],
                [phi_bound[j, k], phi_bound[j, k + 1]],
                [theta_bound[j, k], theta_bound[j + 1, k]]
            )
            A = A + Atmp
            B = B + Btmp

    # Partial hemisphere, -y direction integration range
    phi_vals = _colon(np.pi, 2 * np.pi, n_phi)
    theta_vals = _colon(0, (np.pi / 2 - theta_n) / 2, n_theta)
    phi_bound, theta_bound = np.meshgrid(phi_vals, theta_vals)

    alp_, bet_ = calc_coeff(phi_bound, theta_bound, theta_n, sat_normal, lam, c_coef)

    for j in range(phi_bound.shape[0] - 1):
        for k in range(phi_bound.shape[1] - 1):
            Atmp, Btmp = analytic_sol_ct(
                theta_n, alp_[j, k], bet_[j, k],
                [phi_bound[j, k], phi_bound[j, k + 1]],
                [theta_bound[j, k], theta_bound[j + 1, k]]
            )
            A = A + Atmp
            B = B + Btmp

    area = sat.area[0] if hasattr(sat.area, '__len__') else sat.area
    srp = sunlit_flag * coeff * area * M * (A + B)

    return srp


def srp_approx_ct2(sat: SatelliteModel, theta_n: np.ndarray, sun_b: np.ndarray,
                   d: float, const: Any, n_theta: int = 6, n_phi: int = 6) -> np.ndarray:
    """
    # Approximating specular term of SRP with Cook-Torrance model (multiple facets)

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    theta_n : np.ndarray
        angle between sun vector and facet's normal vector, nFacet x 1
    sun_b : np.ndarray
        sun vector from satellite to Sun expressed with body-fixed frame, 1x3 vector
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        constants for orbital propagation
    n_theta : int, optional
        number of theta divisions (default: 6)
    n_phi : int, optional
        number of phi divisions (default: 6)

    Returns
    -------
    srp : np.ndarray
        approximated SRP with Cook-Torrance model expressed with Sun-fixed frame, 1x3

    Notes
    -----
    Vectorized version for multiple facets

    References
    ----------
    Analytic Approximation of High-Fidelity Solar Radiation Pressure.

    Revisions
    ---------
    20200915  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_approx_ct
    """
    from ..conversion import km2au

    theta_n = np.atleast_1d(theta_n)
    n_facet = len(theta_n)

    # Coefficient
    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au ** 2

    sun_b = np.asarray(sun_b).flatten()

    # Normal vectors, nFacet x 3 matrix
    sat_normals = np.column_stack([np.zeros(n_facet), np.sin(theta_n), np.cos(theta_n)])

    # Perfect mirror-like reflection vector
    r_ref = 2 * (sat_normals @ sun_b)[:, np.newaxis] * sat_normals - sun_b

    # Create temporary satellite for ct_m2
    class TempSat:
        pass

    temp_sat = TempSat()
    temp_sat.normal = sat_normals
    temp_sat.F0 = sat.F0

    M = ct_m2(temp_sat, r_ref, sun_b)

    lam = 2 / sat.mCT ** 2
    mu = 1
    sunlit_flag = 1  # Assuming all facets are sunlit

    srp_tmp = np.zeros((n_facet, 3))

    for i in range(n_facet):
        sat_n = sat_normals[i, :]
        lam_tmp = lam[i]

        # Quarter-sphere, +y direction
        phi_vals = _colon(0, np.pi, n_phi)
        theta_vals = _colon(0, (np.pi / 2 + theta_n[i]) / 2, n_theta)
        phi_bound, theta_bound = np.meshgrid(phi_vals, theta_vals)

        alp_, bet_ = calc_coeff(phi_bound, theta_bound, theta_n[i], sat_n, lam_tmp, mu)

        A = np.zeros(3)
        B = np.zeros(3)
        for j in range(phi_bound.shape[0] - 1):
            for k in range(phi_bound.shape[1] - 1):
                Atmp, Btmp = analytic_sol_ct(
                    theta_n[i], alp_[j, k], bet_[j, k],
                    [phi_bound[j, k], phi_bound[j, k + 1]],
                    [theta_bound[j, k], theta_bound[j + 1, k]]
                )
                A = A + Atmp
                B = B + Btmp

        # Partial hemisphere, -y direction
        phi_vals = _colon(np.pi, 2 * np.pi, n_phi)
        theta_vals = _colon(0, (np.pi / 2 - theta_n[i]) / 2, n_theta)
        phi_bound, theta_bound = np.meshgrid(phi_vals, theta_vals)

        alp_, bet_ = calc_coeff(phi_bound, theta_bound, theta_n[i], sat_n, lam_tmp, mu)

        for j in range(phi_bound.shape[0] - 1):
            for k in range(phi_bound.shape[1] - 1):
                Atmp, Btmp = analytic_sol_ct(
                    theta_n[i], alp_[j, k], bet_[j, k],
                    [phi_bound[j, k], phi_bound[j, k + 1]],
                    [theta_bound[j, k], theta_bound[j + 1, k]]
                )
                A = A + Atmp
                B = B + Btmp

        srp_tmp[i, :] = sunlit_flag * coeff * sat.area[i] * M[i] * (A + B)

    srp = np.sum(srp_tmp, axis=0)

    return srp


# %[appendix]{"version":"1.0"}
