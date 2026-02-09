"""
Light curve calculation models
Python conversion from yMATLAB/lightcurves/
"""

import numpy as np
from ..object import SatelliteModel


def lc(sat: SatelliteModel, scalar: int, q: np.ndarray,
       sat_pos: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
       nu: np.ndarray, brdf: str = 'simple') -> tuple[np.ndarray, np.ndarray]:
    """
    # Calculating light curves

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    scalar : int
        quaternion convention (0 or 4)
        scalar == 0: q = [q0, q1, q2, q3]^T = [cos(theta/2), e^T*sin(theta/2)]^T
        scalar == 4: q = [q1, q2, q3, q4]^T = [e^T*sin(theta/2), cos(theta/2)]^T
    q : np.ndarray
        quaternions, N x 4 matrix
    sat_pos : np.ndarray
        satellite position vector in inertial frame, m, N x 3 matrix
    obs_pos : np.ndarray
        observer position vector in inertial frame, m, N x 3 matrix
    sun_pos : np.ndarray
        sun position vector in inertial frame, m, N x 3 matrix
    nu : np.ndarray
        eclipse flag, N x 1 vector (1: sunlit, 0: eclipse)
    brdf : str, optional
        BRDF model used in light curve calculation:
        - 'simple' (default): Lambertian diffuse and mirror-like specular
        - 'AS': Ashikhmin-Shirley model
        - 'CT': Cook-Torrance model

    Returns
    -------
    m : np.ndarray
        relative magnitude of light curves w.r.t. Sun, N x 1 vector
    f_obs : np.ndarray
        observed flux, N x 1 vector

    Notes
    -----
    Main wrapper function for light curve calculation.

    References
    ----------
    NA

    Revisions
    ---------
    20200430  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    read_sc, lc_simple, lc_as, lc_ct, orbit_const
    """
    from ..attitude import q_rotation
    from ..math_utils import norm_row

    sat_pos = np.atleast_2d(sat_pos)
    obs_pos = np.atleast_2d(obs_pos)
    sun_pos = np.atleast_2d(sun_pos)
    q = np.atleast_2d(q)
    nu = np.atleast_1d(nu).flatten()

    # Sun and observer relative position
    sun_rel_dir = sun_pos - sat_pos  # m, relative direction from sat to sun@ECI
    obs_rel_dir = obs_pos - sat_pos  # m, relative from sat to observer

    # Transform to body-fixed frame
    sun_b = q_rotation(norm_row(sun_rel_dir), q, scalar=scalar)  # sun direction (unit) vector@body-fixed frame
    obs_b = q_rotation(norm_row(obs_rel_dir), q, scalar=scalar)  # observer direction (unit) vector@body-fixed frame

    # Calculate relative magnitude of light curves
    if brdf == 'simple':
        sat, _, _ = lc_simple(sat, sun_b, obs_b)
    elif brdf == 'AS':
        sat, _, _, _ = lc_as(sat, sun_b, obs_b)
    elif brdf == 'CT':
        sat, _, _, _ = lc_ct(sat, sun_b, obs_b)
    else:
        raise ValueError(f"Unavailable BRDF model: {brdf}. Use 'simple', 'AS', or 'CT'.")

    f_obs = np.sum(sat.f_obs, axis=0)
    f_obs = f_obs.flatten()  # column vector as time history

    # Consider umbra, penumbra
    f_obs = f_obs * nu

    # Compute magnitude
    obs_dist = np.linalg.norm(obs_rel_dir, axis=1)
    m = -26.7 - 2.5 * np.log10(f_obs / obs_dist**2)

    return m, f_obs


def lc_simple(sat: SatelliteModel, sun_b: np.ndarray, obs_b: np.ndarray,
              thr: float = np.deg2rad(1.0)) -> tuple[SatelliteModel, np.ndarray, np.ndarray]:
    """
    # calculating Lambertian diffusion and Perfect Specularity light curve

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration
    sun_b : np.ndarray
        sun vector from satellite to Sun in body frame, M x 3
    obs_b : np.ndarray
        observer vector from satellite in body frame, M x 3
    thr : float, optional
        threshold for specular lobe, rad (default: 1 degree)

    Returns
    -------
    sat : SatelliteModel
        updated satellite with fObs
    cd : np.ndarray
        diffuse part of BRDF, nFacet x M
    cs : np.ndarray
        specular part of BRDF, nFacet x M

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20251208  y.yoshimura

    See also
    --------
    lc_as, lc_ct, read_sc
    """
    sun_b = np.atleast_2d(sun_b)
    obs_b = np.atleast_2d(obs_b)

    # Normalize
    sun_b = sun_b / np.linalg.norm(sun_b, axis=1, keepdims=True)
    obs_b = obs_b / np.linalg.norm(obs_b, axis=1, keepdims=True)

    # Bisector (half) vector
    h = sun_b + obs_b
    h = h / np.linalg.norm(h, axis=1, keepdims=True)

    # Dot products: nFacet x M
    NS = sat.normal @ sun_b.T
    NV = sat.normal @ obs_b.T
    NH = sat.normal @ h.T

    # Diffuse and specular components
    c_total = (sat.Cd[:, np.newaxis] / np.pi
               + 2.0 * sat.Cs[:, np.newaxis] * (NH >= np.cos(thr)))

    f_obs = c_total * sat.area[:, np.newaxis] * NS * NV

    # Visibility check
    visible = (NS > 0) & (NV > 0)
    cd = (sat.Cd[:, np.newaxis] / np.pi) * visible
    cs = 2.0 * sat.Cs[:, np.newaxis] * (NH >= np.cos(thr)) * visible
    f_obs = f_obs * visible

    # Store in satellite model (as attribute if needed)
    # sat.f_obs = f_obs

    return sat, cd, cs


def lc_as(sat: SatelliteModel, sun_b: np.ndarray, obs_b: np.ndarray
          ) -> tuple[SatelliteModel, np.ndarray, np.ndarray, np.ndarray]:
    """
    # Calculating light curves using Ashikhmin-Shirley model

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc, N facets
    sun_b : np.ndarray
        sun vector from satellite, unit vector, M x 3 matrix
    obs_b : np.ndarray
        observer vector from satellite to observer, unit vector, M x 3 matrix

    Returns
    -------
    sat : SatelliteModel
        sat.f_obs is added, N x M
    cd : np.ndarray
        diffuse part of light curves, N x M matrix
    cs : np.ndarray
        specular part of light curves, N x M matrix
    D : np.ndarray
        NDF distribution function, N x M matrix

    Notes
    -----
    Calculated in body-fixed frame (normal vector is not necessarily [0, 0, 1]^T)

    References
    ----------
    Ashikhmin, Michael, & Shirley, Peter. "An Anisotropic Phong BRDF Model."
    Journal of graphics tools, vol. 5, no. 2, 2000, pp. 25-32.

    Revisions
    ---------
    20200430  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    read_sc, lc_simple, lc_ct
    """
    sun_b = np.atleast_2d(sun_b)
    obs_b = np.atleast_2d(obs_b)

    # Normalize
    sun_b = sun_b / np.linalg.norm(sun_b, axis=1, keepdims=True)
    obs_b = obs_b / np.linalg.norm(obs_b, axis=1, keepdims=True)

    # Bisector
    h = sun_b + obs_b
    h = h / np.linalg.norm(h, axis=1, keepdims=True)

    # Pre-calculation
    NS = sat.normal @ sun_b.T  # nFacet x M
    NV = sat.normal @ obs_b.T  # nFacet x M
    VH = np.sum(obs_b * h, axis=1)  # M,
    NH = sat.normal @ h.T  # nFacet x M
    HU = sat.uu @ h.T  # nFacet x M
    HV = sat.uv @ h.T  # nFacet x M

    # Diffuse
    cd = (28 / 23 * sat.Cd[:, np.newaxis] / np.pi * (1 - sat.F0[:, np.newaxis])
          * (1 - (1 - NS / 2) ** 5) * (1 - (1 - NV / 2) ** 5))

    # Specular
    F = sat.F0[:, np.newaxis] + (1 - sat.F0[:, np.newaxis]) * (1 - VH) ** 5
    M_val = (np.sqrt((sat.nu[:, np.newaxis] + 1) * (sat.nv[:, np.newaxis] + 1))
             / 8 / np.pi * F / VH / np.maximum(NS, NV))
    M_val = np.where(np.isinf(M_val), 0, M_val)

    # Zero out where NS < 0 or NV < 0 to prevent complex numbers
    NH_safe = np.where((NS < 0) | (NV < 0), 0, NH)
    denom = 1 - NH_safe ** 2
    denom = np.where(denom == 0, 1e-10, denom)  # Avoid division by zero

    exp_val = (sat.nu[:, np.newaxis] * HU ** 2 + sat.nv[:, np.newaxis] * HV ** 2) / denom
    D = NH_safe ** exp_val
    cs = M_val * D

    # Total
    c_total = cd + cs
    tmp = c_total * sat.area[:, np.newaxis] * NS * NV

    # Visibility check
    visible = (NS > 0) & (NV > 0)
    cd = cd * visible
    cs = cs * visible
    sat.f_obs = tmp * visible

    return sat, cd, cs, D


def lc_ct(sat: SatelliteModel, sun_b: np.ndarray, obs_b: np.ndarray,
          ndf: str = 'Beckmann') -> tuple[SatelliteModel, np.ndarray, np.ndarray, np.ndarray]:
    """
    # Calculating Cook-Torrance model light curves

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, M x 3 matrix
    obs_b : np.ndarray
        observer from satellite in body-fixed frame, M x 3 matrix
    ndf : str, optional
        NDF definition: 'Beckmann' (default) or 'Gauss'

    Returns
    -------
    sat : SatelliteModel
        sat.f_obs is added, nFacet x M
    cd : np.ndarray
        diffuse part of BRDF, nFacet x M vector
    cs : np.ndarray
        specular part of BRDF, nFacet x M vector
    D : np.ndarray
        normal distribution function of BRDF, nFacet x M vector

    Notes
    -----
    Cook-Torrance model:
    c_d = rho_d / pi
    c_s = DGF / (4 * (n^T s)(n^T v))

    References
    ----------
    Cook, R. L., & Torrance, K. E. (1982). A reflectance model for computer graphics.
    ACM Transactions on Graphics (TOG), 1, 7-24.

    Revisions
    ---------
    20251208  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    lc_as, read_sc
    """
    sun_b = np.atleast_2d(sun_b)
    obs_b = np.atleast_2d(obs_b)
    M = sun_b.shape[0]

    # Normalize
    sun_b = sun_b / np.linalg.norm(sun_b, axis=1, keepdims=True)
    obs_b = obs_b / np.linalg.norm(obs_b, axis=1, keepdims=True)

    h = sun_b + obs_b
    h = h / np.linalg.norm(h, axis=1, keepdims=True)

    NS = sat.normal @ sun_b.T  # nFacet x M
    NH = sat.normal @ h.T  # nFacet x M
    NV = sat.normal @ obs_b.T  # nFacet x M
    VH = np.sum(obs_b * h, axis=1)  # M,

    theta_h = np.arccos(NH)  # nFacet x M

    # Diffuse
    cd = np.tile(sat.Cd[:, np.newaxis] / np.pi, (1, M))

    # Specular
    nest = (1 + np.sqrt(sat.F0[:, np.newaxis])) / (1 - np.sqrt(sat.F0[:, np.newaxis]))
    g = np.sqrt(nest ** 2 + VH ** 2 - 1)

    if ndf == 'Beckmann':
        # Beckmann distribution
        tan_theta_h = np.tan(theta_h)
        cos_theta_h = np.cos(theta_h)
        D = np.exp(-(tan_theta_h / sat.mCT[:, np.newaxis]) ** 2)
        D = D / np.pi / sat.mCT[:, np.newaxis] ** 2 / cos_theta_h ** 4
    elif ndf == 'Gauss':
        # Gaussian distribution
        D = np.exp(-(theta_h / sat.mCT[:, np.newaxis]) ** 2)
    else:
        raise ValueError("Set proper NDF option: 'Beckmann' or 'Gauss'")

    temp1 = 2 * NH * NV / VH
    temp2 = 2 * NH * NS / VH

    G = np.minimum(1, temp1)
    G = np.minimum(G, temp2)

    temp1 = (g - VH) ** 2 / 2 / (g + VH) ** 2
    temp2 = 1 + (VH * (g + VH) - 1) ** 2 / (VH * (g - VH) + 1) ** 2
    F = temp1 * temp2

    cs = D * G * F / NS / NV / 4

    tmp = (cd + cs) * sat.area[:, np.newaxis] * NS * NV

    # Visibility check
    visible = (NS > 0) & (NV > 0)
    cd = cd * visible
    cs = cs * visible
    sat.f_obs = tmp * visible

    return sat, cd, cs, D


# %[appendix]{"version":"1.0"}


def mag(f_obs: float | np.ndarray, d_sun: float, d_obs: float,
        sun_mag: float = -26.74) -> float | np.ndarray:
    """
    # Calculate apparent magnitude from observed flux

    Parameters
    ----------
    f_obs : float or np.ndarray
        observed flux
    d_sun : float
        distance from satellite to Sun, m
    d_obs : float
        distance from satellite to observer, m
    sun_mag : float, optional
        apparent magnitude of the Sun (default: -26.74)

    Returns
    -------
    m : float or np.ndarray
        apparent magnitude

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
    mag_inv
    """
    AU_M = 149597870700.0  # m

    m = sun_mag - 2.5 * np.log10(f_obs * (AU_M / d_sun) ** 2 * (1.0 / d_obs) ** 2)

    return m


# %[appendix]{"version":"1.0"}


def mag_inv(m: float | np.ndarray, d_sun: float, d_obs: float,
            sun_mag: float = -26.74) -> float | np.ndarray:
    """
    # Calculate observed flux from apparent magnitude

    Parameters
    ----------
    m : float or np.ndarray
        apparent magnitude
    d_sun : float
        distance from satellite to Sun, m
    d_obs : float
        distance from satellite to observer, m
    sun_mag : float, optional
        apparent magnitude of the Sun (default: -26.74)

    Returns
    -------
    f_obs : float or np.ndarray
        observed flux

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
    mag
    """
    AU_M = 149597870700.0  # m

    f_obs = 10 ** ((sun_mag - m) / 2.5) * (d_sun / AU_M) ** 2 * d_obs ** 2

    return f_obs


# %[appendix]{"version":"1.0"}
