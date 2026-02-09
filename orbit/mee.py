"""
Modified Equinoctial Elements (MEE) and Gauss Variational Equations
Python conversion from yMATLAB/orbit/mee.m, coe2mee.m, mee2coe.m, gve.m
"""

import numpy as np
from typing import Union
from dataclasses import dataclass, field
from .kepler import mean_anomaly, true_anomaly


@dataclass
class OrbitalElements:
    """
    Orbital elements container supporting both COE and MEE
    """
    # Classical orbital elements
    a: float = 0.0       # semi-major axis
    e: float = 0.0       # eccentricity
    inc: float = 0.0     # inclination
    raan: float = 0.0    # right ascension of ascending node
    w: float = 0.0       # argument of perigee
    nu: float = 0.0      # true anomaly
    M: float = 0.0       # mean anomaly

    # Modified equinoctial elements
    p_: float = 0.0      # semi-latus rectum
    f_: float = 0.0      # e*cos(w+raan)
    g_: float = 0.0      # e*sin(w+raan)
    h_: float = 0.0      # tan(inc/2)*cos(raan)
    k_: float = 0.0      # tan(inc/2)*sin(raan)
    L_: float = 0.0      # w + nu + raan (true longitude)

    # Derived quantities
    p: float = 0.0       # semi-latus rectum (COE)
    h: float = 0.0       # angular momentum magnitude
    n: float = 0.0       # mean motion
    u: float = 0.0       # true argument of latitude
    uM: float = 0.0      # mean argument of latitude
    r: float = 0.0       # radius

    # Position and velocity
    rVec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vVec: np.ndarray = field(default_factory=lambda: np.zeros(3))


def coe2mee(oe: OrbitalElements) -> OrbitalElements:
    """
    # Calculate modified equinoctial elements from classical orbital elements

    Parameters
    ----------
    oe : OrbitalElements
        orbital elements with COE filled

    Returns
    -------
    oe : OrbitalElements
        orbital elements with MEE filled

    Notes
    -----
    Modified equinoctial elements:
    oe_MEE = [p, f, g, h, k, L] =
    [a(1-e^2), e*cos(w+Ω), e*sin(w+Ω), tan(i/2)*cos(Ω), tan(i/2)*sin(Ω), w+Ω+ν]

    Classical orbital elements:
    oe = [a, e, i, Ω, w, ν (or M)]

    References
    ----------
    Vallado, D.A., Fundamentals of Astrodynamics and Applications

    Revisions
    ---------
    NA

    See also
    --------
    mee2coe
    """
    oe.p_ = oe.a * (1 - oe.e ** 2)
    oe.f_ = oe.e * np.cos(oe.w + oe.raan)
    oe.g_ = oe.e * np.sin(oe.w + oe.raan)
    oe.h_ = np.tan(oe.inc / 2) * np.cos(oe.raan)
    oe.k_ = np.tan(oe.inc / 2) * np.sin(oe.raan)
    oe.L_ = oe.w + oe.nu + oe.raan

    oe.L_ = np.mod(oe.L_, 2 * np.pi)

    return oe


def mee2coe(oe: OrbitalElements) -> OrbitalElements:
    """
    # Calculate classical orbital elements from modified equinoctial elements

    Parameters
    ----------
    oe : OrbitalElements
        orbital elements with MEE filled

    Returns
    -------
    oe : OrbitalElements
        orbital elements with COE filled

    Notes
    -----
    Modified equinoctial elements:
    oe_MEE = [p, f, g, h, k, L]

    Classical orbital elements:
    oe = [a, e, i, Ω, w, ν]

    References
    ----------
    Vallado, D.A., Fundamentals of Astrodynamics and Applications

    Revisions
    ---------
    NA

    See also
    --------
    coe2mee
    """
    oe.a = oe.p_ / (1 - oe.f_ ** 2 - oe.g_ ** 2)
    oe.e = np.sqrt(oe.f_ ** 2 + oe.g_ ** 2)
    oe.inc = np.arctan2(2 * np.sqrt(oe.h_ ** 2 + oe.k_ ** 2), 1 - oe.h_ ** 2 - oe.k_ ** 2)
    oe.raan = np.arctan2(oe.k_, oe.h_)
    oe.w = np.arctan2(oe.g_ * oe.h_ - oe.f_ * oe.k_, oe.f_ * oe.h_ + oe.g_ * oe.k_)
    oe.nu = oe.L_ - oe.raan - oe.w

    oe.raan = np.mod(oe.raan, 2 * np.pi)
    oe.w = np.mod(oe.w, 2 * np.pi)
    oe.nu = np.mod(oe.nu, 2 * np.pi)

    return oe


def calc_orbital_state(oe: OrbitalElements, mu: float) -> OrbitalElements:
    """
    # Calculate derived orbital parameters

    Parameters
    ----------
    oe : OrbitalElements
        orbital elements
    mu : float
        gravitational parameter

    Returns
    -------
    oe : OrbitalElements
        orbital elements with derived quantities filled

    Revisions
    ---------
    NA
    """
    from .orbital_elements import oe2rv

    oe.p = oe.a * (1 - oe.e ** 2)  # semi-latus rectum
    oe.h = np.sqrt(mu * oe.p)      # orbital angular momentum
    oe.n = np.sqrt(mu / oe.a ** 3) # mean motion
    oe.u = oe.w + oe.nu            # true argument of latitude
    oe.M = mean_anomaly(oe.e, oe.nu)
    oe.uM = oe.w + oe.M            # mean argument of latitude
    oe.r = oe.p / (1.0 + oe.e * np.cos(oe.nu))  # radius

    oe.u = np.mod(oe.u, 2 * np.pi)
    oe.M = np.mod(oe.M, 2 * np.pi)
    oe.uM = np.mod(oe.uM, 2 * np.pi)

    # Calculate position and velocity vectors
    oe_arr = np.array([oe.a, oe.e, oe.inc, oe.raan, oe.w, oe.nu])
    r_tmp, v_tmp = oe2rv(oe_arr, 1, mu)

    oe.rVec = r_tmp.flatten()
    oe.vVec = v_tmp.flatten()

    return oe


def mee_derivatives(oe: OrbitalElements, a_rtn: np.ndarray, mu: float) -> np.ndarray:
    """
    # Calculate MEE derivatives using Variational Equations

    Parameters
    ----------
    oe : OrbitalElements
        orbital elements with MEE filled
    a_rtn : np.ndarray
        perturbation acceleration in RTN frame, 3x1 [aR, aT, aN]
    mu : float
        gravitational parameter

    Returns
    -------
    d_oe_dt : np.ndarray
        time derivatives [dp/dt, df/dt, dg/dt, dh/dt, dk/dt, dL/dt], 6x1

    Notes
    -----
    dp/dt = (2p/w) * sqrt(p/μ) * aT
    df/dt = sqrt(p/μ) * [aR*sin(L) + ((w+1)*cos(L)+f)/w * aT - g*(h*sin(L)-k*cos(L))/w * aN]
    dg/dt = sqrt(p/μ) * [-aR*cos(L) + ((w+1)*sin(L)+g)/w * aT + f*(h*sin(L)-k*cos(L))/w * aN]
    dh/dt = sqrt(p/μ) * s²*aN*cos(L) / (2w)
    dk/dt = sqrt(p/μ) * s²*aN*sin(L) / (2w)
    dL/dt = sqrt(μp) * (w/p)² + sqrt(p/μ) * (h*sin(L)-k*cos(L))/w * aN

    References
    ----------
    Vallado, D.A., Fundamentals of Astrodynamics and Applications

    Revisions
    ---------
    NA
    """
    p = oe.p_
    f = oe.f_
    g = oe.g_
    h = oe.h_
    k = oe.k_
    L = oe.L_

    w = 1 + f * np.cos(L) + g * np.sin(L)
    s2 = 1 + h ** 2 + k ** 2

    sqrt_p_mu = np.sqrt(p / mu)
    hsinL_kcosL = h * np.sin(L) - k * np.cos(L)

    d_oe_dt = np.array([
        2 * p / w * sqrt_p_mu * a_rtn[1],
        sqrt_p_mu * (a_rtn[0] * np.sin(L) + ((w + 1) * np.cos(L) + f) * a_rtn[1] / w
                     - g * hsinL_kcosL * a_rtn[2] / w),
        sqrt_p_mu * (-a_rtn[0] * np.cos(L) + ((w + 1) * np.sin(L) + g) * a_rtn[1] / w
                     + f * hsinL_kcosL * a_rtn[2] / w),
        sqrt_p_mu * s2 * a_rtn[2] * np.cos(L) / 2 / w,
        sqrt_p_mu * s2 * a_rtn[2] * np.sin(L) / 2 / w,
        np.sqrt(mu * p) * (w / p) ** 2 + sqrt_p_mu * hsinL_kcosL * a_rtn[2] / w
    ])

    return d_oe_dt


def mee(oe: OrbitalElements | dict, a_rtn: np.ndarray, mu: float) -> np.ndarray:
    """
    # MATLAB-compatible wrapper for MEE derivatives (MATLAB: mee)

    Parameters
    ----------
    oe : OrbitalElements or dict
        orbital elements with MEE fields (p_, f_, g_, h_, k_, L_)
    a_rtn : np.ndarray
        perturbation acceleration in RTN frame, 3x1 [aR, aT, aN]
    mu : float
        gravitational parameter

    Returns
    -------
    d_oe_dt : np.ndarray
        time derivatives [dp/dt, df/dt, dg/dt, dh/dt, dk/dt, dL/dt], 6x1
    """
    if isinstance(oe, OrbitalElements):
        oe_local = oe
    elif isinstance(oe, dict):
        oe_local = OrbitalElements(
            p_=oe.get('p_', 0.0),
            f_=oe.get('f_', 0.0),
            g_=oe.get('g_', 0.0),
            h_=oe.get('h_', 0.0),
            k_=oe.get('k_', 0.0),
            L_=oe.get('L_', 0.0),
        )
    elif all(hasattr(oe, attr) for attr in ['p_', 'f_', 'g_', 'h_', 'k_', 'L_']):
        oe_local = oe
    else:
        raise TypeError("mee expects OrbitalElements or dict with p_, f_, g_, h_, k_, L_")

    return mee_derivatives(oe_local, np.asarray(a_rtn).flatten(), mu)


def gve(oe: OrbitalElements, a_rtn: np.ndarray, anomaly_flag: int, mu: float) -> np.ndarray:
    """
    # Calculate classical orbital element derivatives using Gauss Variational Equations

    Parameters
    ----------
    oe : OrbitalElements
        orbital elements
    a_rtn : np.ndarray
        perturbation acceleration in RTN frame, 3x1 [aR, aT, aN]
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly
    mu : float
        gravitational parameter

    Returns
    -------
    d_oe_dt : np.ndarray
        time derivatives [da/dt, de/dt, di/dt, dΩ/dt, dω/dt, dν/dt (or dM/dt)], 6x1

    Notes
    -----
    da/dt = 2/(n*sqrt(1-e²)) * [e*sin(ν)*aR + (1+e*cos(ν))*aT]
    de/dt = sqrt(1-e²)/(na) * [aR*sin(ν) + (cos(ν) + (e+cos(ν))/(1+e*cos(ν)))*aT]
    di/dt = r*cos(u) / (na²*sqrt(1-e²)) * aN
    dΩ/dt = r*sin(u) / (na²*sqrt(1-e²)*sin(i)) * aN
    dω/dt = -sqrt(1-e²)/(nae) * [aR*cos(ν) - (sin(ν) + sin(ν)/(1+e*cos(ν)))*aT] - dΩ/dt*cos(i)
    dν/dt = h/r² - dω/dt - dΩ/dt*cos(i)

    References
    ----------
    Vallado, D.A., Fundamentals of Astrodynamics and Applications

    Revisions
    ---------
    NA

    See also
    --------
    mee_derivatives
    """
    oe = calc_orbital_state(oe, mu)
    r = oe.r
    u = oe.u
    n = oe.n
    p = oe.p
    h = oe.h

    a = oe.a
    e = oe.e
    inc = oe.inc
    nu = oe.nu

    sqrt_1_e2 = np.sqrt(1 - e ** 2)

    # Calculate RAAN rate
    d_raan_dt = r * np.sin(u) / n / a ** 2 / sqrt_1_e2 / np.sin(inc) * a_rtn[2]

    # Calculate argument of perigee rate
    d_w_dt = (-sqrt_1_e2 / n / a / e * (np.cos(nu) * a_rtn[0]
              - (np.sin(nu) + np.sin(nu) / (1 + e * np.cos(nu))) * a_rtn[1])
              - d_raan_dt * np.cos(inc))

    # Calculate anomaly rate
    if anomaly_flag == 1:  # True anomaly
        tmp = h / r ** 2 - d_w_dt - d_raan_dt * np.cos(inc)
    else:  # Mean anomaly
        tmp = (n + 1 / n / a ** 2 / e * ((p * np.cos(nu) - 2 * e * r) * a_rtn[0]
               - (p + r) * np.sin(nu) * a_rtn[1]))

    # Assemble Gauss Variational Equations
    d_oe_dt = np.array([
        2 / n / sqrt_1_e2 * (e * np.sin(nu) * a_rtn[0] + (1 + e * np.cos(nu)) * a_rtn[1]),
        sqrt_1_e2 / n / a * (np.sin(nu) * a_rtn[0]
                             + (np.cos(nu) + (e + np.cos(nu)) / (1 + e * np.cos(nu))) * a_rtn[1]),
        r * np.cos(u) / (n * a ** 2 * sqrt_1_e2) * a_rtn[2],
        d_raan_dt,
        d_w_dt,
        tmp
    ])

    return d_oe_dt


def mean2osc(n_rev: float, e: float, inc: float, raan: float, w: float,
             M: float, const) -> np.ndarray:
    """
    # Converting mean orbital elements to osculating orbital elements

    Parameters
    ----------
    n_rev : float
        mean motion, rev/day
    e : float
        eccentricity
    inc : float
        inclination, rad
    raan : float
        argument of ascending node, rad
    w : float
        argument of perihelion, rad
    M : float
        mean anomaly, rad
    const : OrbitalConstants
        constant parameters

    Returns
    -------
    osc : np.ndarray
        osculating orbital elements:
        [a_osc, e_osc, i_osc, Ω_osc, ω_osc, f_osc, M_osc, r_osc, dr_osc, p_osc, u_osc]

    Notes
    -----
    平均軌道要素を接触軌道要素へ変換

    References
    ----------
    David A. Vallado, "Fundamentals of Astrodynamics and Applications, 4th edition, pp.708-709.

    Revisions
    ---------
    20210531  y.yoshimura

    See also
    --------
    orbit_const, true_anomaly
    """
    # Mean values
    n_sec = 2 * np.pi * n_rev / (24 * 60 * 60)  # mean motion, rad/s
    a = (const.GE / n_sec ** 2) ** (1 / 3)  # semi-major axis, km
    q = a * (1 - e)
    p = q * (1 + e)

    f, _ = true_anomaly(a, e, M)
    u = w + f  # argument of latitude

    r = p / (1 + e * np.cos(f))
    dr_mean = np.sqrt(const.GE / p) * e * np.sin(f)  # radial velocity

    coef = const.J2 * const.RE ** 2

    # Short-period variations
    # Δi_SP, Δp_SP, ΔΩ_SP
    tmp = (3.0 * np.cos(2.0 * u) + 3.0 * e * np.cos(2.0 * w + f)
           + e * np.cos(2.0 * w + 3.0 * f))

    di = tmp * coef * np.sin(inc) * np.cos(inc) / 4 / p ** 2
    dp = tmp * coef * np.sin(inc) ** 2 / 2 / p

    d_raan = (6.0 * (f - M + e * np.sin(f)) - 3.0 * np.sin(2.0 * u)
              - 3.0 * e * np.sin(2.0 * w + f) - e * np.sin(2.0 * w + 3.0 * f))
    d_raan = -d_raan * coef * np.cos(inc) / 4 / p ** 2

    # Δr_SP, Δṙ_SP, Δu_SP
    dr = ((3 * np.cos(inc) ** 2 - 1) * (2 * np.sqrt(1 - e ** 2) / (1 + e * np.cos(f))
          + e * np.cos(f) / (1 + np.sqrt(1 - e ** 2)) + 1)
          - np.sin(inc) ** 2 * np.cos(2 * u))
    dr = -dr * coef / 4 / p

    ddr = ((3 * np.cos(inc) ** 2 - 1) * e * np.sin(f) * (np.sqrt(1 - e ** 2)
           + (1 + e * np.cos(f)) ** 2 / (1 + np.sqrt(1 - e ** 2)))
           - 2 * np.sin(inc) ** 2 * (1 - e * np.cos(f)) ** 2 * np.sin(2 * u))
    ddr = ddr * coef * np.sqrt(const.GE) / 4 / p ** (5 / 2)

    du = ((6 - 30 * np.cos(inc) ** 2) * (f - M) + 4 * e * np.sin(f) * ((1 - 6 * np.cos(inc) ** 2)
          + (1 - 3 * np.cos(inc) ** 2) / (1 + np.sqrt(1 - e ** 2)))
          + (1 - 3 * np.cos(inc) ** 2) / (1 + np.sqrt(1 - e ** 2)) * e ** 2 * np.sin(2 * f)
          + (5 * np.cos(inc) ** 2 - 2) * 2 * e * np.sin(f + 2 * w)
          + (7 * np.cos(inc) ** 2 - 1) * np.sin(2 * u)
          + 2 * np.cos(inc) ** 2 * e * np.sin(3 * f + 2 * w))
    du = du * coef / 8 / p ** 2

    # Correct mean values
    r_osc = r + dr
    dr_osc = dr_mean + ddr
    p_osc = p + dp

    A = p_osc / r_osc - 1
    B = np.sqrt(p_osc / const.GE) * dr_osc
    e_osc = np.sqrt(A ** 2 + B ** 2)
    a_osc = p_osc / (1 - e_osc ** 2)
    i_osc = inc + di
    raan_osc = raan + d_raan
    u_osc = u + du
    f_osc = np.arctan2(B, A)
    w_osc = u_osc - f_osc

    # Eccentric anomaly
    ea = 2.0 * np.arctan(np.sqrt((1.0 - e_osc) / (1.0 + e_osc)) * np.tan(0.5 * f_osc))

    # Mean anomaly
    M_osc = ea - e_osc * np.sin(ea)

    # Wrap to [0, 2π]
    raan_osc = np.mod(raan_osc, 2 * np.pi)
    w_osc = np.mod(w_osc, 2 * np.pi)
    f_osc = np.mod(f_osc, 2 * np.pi)
    M_osc = np.mod(M_osc, 2 * np.pi)

    osc = np.array([a_osc, e_osc, i_osc, raan_osc, w_osc, f_osc, M_osc, r_osc, dr_osc, p_osc, u_osc])

    return osc


# %[appendix]{"version":"1.0"}
