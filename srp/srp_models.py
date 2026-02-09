"""
Solar Radiation Pressure (SRP) models
Python conversion from yMATLAB/srp/
"""

import numpy as np
from ..object import SatelliteModel


# Physical constants
S0 = 1361.0  # Solar constant, W/m^2 (at 1 AU)
C_LIGHT = 299792458.0  # Speed of light, m/s
AU_M = 149597870700.0  # Astronomical Unit, m


def srp_simple(sat: SatelliteModel, sun_b: np.ndarray, d: float
               ) -> tuple[SatelliteModel, np.ndarray, np.ndarray]:
    """
    # calculating solar radiation pressure (SRP) with Lambertian and specular model

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1 x 3 unit vector
    d : float
        distance between satellite and Sun, m

    Returns
    -------
    sat : SatelliteModel
        updated satellite with force and torque
    srp_cd_out : np.ndarray
        total diffuse SRP force in body frame, N, 1 x 3
    srp_cs_out : np.ndarray
        total specular SRP force in body frame, N, 1 x 3

    Notes
    -----
    Uses perfect Lambertian reflection and perfect specular reflection model.

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    read_sc
    """
    Bf = 2.0 / 3.0
    kappa = np.zeros_like(sat.Ca)  # thermal emissivity (default: 0)

    d_au = d / AU_M
    sun_b = np.asarray(sun_b).flatten()
    sun_b = sun_b / np.linalg.norm(sun_b)

    coeff = -S0 / C_LIGHT / (d_au ** 2)

    # Shadowing flag
    ns = sat.normal @ sun_b  # n x 1
    sunlit_flag = (ns > 0).astype(float)

    # Diffuse
    srp_cd = (2.0 / 3.0) * sat.Cd[:, np.newaxis] * sat.normal

    # Specular
    r_ref = 2 * ns[:, np.newaxis] * sat.normal - sun_b
    srp_cs = sat.Cs[:, np.newaxis] * r_ref

    # Total force
    tmp = (ns[:, np.newaxis] * (sat.Ca[:, np.newaxis] + sat.Cd[:, np.newaxis]) * sun_b
           + ns[:, np.newaxis] * (Bf * sat.Cd[:, np.newaxis] + kappa[:, np.newaxis] * sat.Ca[:, np.newaxis]
                                   + 2.0 * sat.Cs[:, np.newaxis] * ns[:, np.newaxis]) * sat.normal)

    sat.force = sunlit_flag[:, np.newaxis] * coeff * sat.area[:, np.newaxis] * tmp
    sat.torque = np.cross(sat.pos, sat.force)

    # Output: diffuse and specular SRP components
    tmp_cd = coeff * sat.area[:, np.newaxis] * ns[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cd
    srp_cd_out = np.sum(tmp_cd, axis=0)

    tmp_cs = coeff * sat.area[:, np.newaxis] * ns[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cs
    srp_cs_out = np.sum(tmp_cs, axis=0)

    return sat, srp_cd_out, srp_cs_out


def srp_as(sat: SatelliteModel, sun_b: np.ndarray, d: float, const,
           n_mc: int = 10000) -> tuple[SatelliteModel, np.ndarray, np.ndarray]:
    """
    # Numerically calculating SRP with Ashikhmin-Shirley model

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1 x 3
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        orbital constants
    n_mc : int, optional
        number of sampling for Monte-Carlo integration (default: 10^4)

    Returns
    -------
    sat : SatelliteModel
        updated satellite with force and torque at each facet
    srp_cd_out : np.ndarray
        total diffuse part of SRP, N, 1 x 3
    srp_cs_out : np.ndarray
        total specular part of SRP, N, 1 x 3

    Notes
    -----
    SRP is written as:
    -S0/(c*r_AU^2) * A * (n^T s) * [s + integral(fr*(n^T v)*sin(theta_r)*v d_theta_r d_phi_r)]

    References
    ----------
    Ashikhmin, Michael, & Shirley, Peter. "An Anisotropic Phong BRDF Model."
    Journal of graphics tools, vol. 5, no. 2, 2000, pp. 25-32.

    Revisions
    ---------
    y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_simple, read_sc, orbit_const
    """
    from ..conversion import km2au
    from ..attitude import q_rotation, q_inv

    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au ** 2

    sun_b = np.asarray(sun_b).flatten()
    sun_b = sun_b / np.linalg.norm(sun_b)

    NS = sat.normal @ sun_b  # n_facet x 1
    n_facet = sat.normal.shape[0]

    # Diffuse (analytic)
    cd1 = 28 / 23 * sat.Cd / np.pi * (1 - sat.F0) * (1 - (1 - NS / 2) ** 5)
    srp_cd = np.zeros((n_facet, 3))
    srp_cd[:, 2] = cd1 * 1573 / 2688 * np.pi

    # Specular (numerical)
    # Transform to local frame (normal vector is along z-axis)
    sun_b_rep = np.tile(sun_b, (n_facet, 1))
    s_local = q_rotation(sun_b_rep, sat.qlb, scalar=4)

    # Importance sampling
    u1 = np.random.rand(1, n_mc)
    u2 = np.random.rand(1, n_mc)

    def phi_fun(u, nu, nv):
        return np.arctan(np.sqrt((nu + 1) / (nv + 1)) * np.tan(np.pi * u / 2))

    ind1 = u1 < 0.25
    ind2 = (u1 >= 0.25) & (u1 < 0.5)
    ind3 = (u1 >= 0.5) & (u1 < 0.75)
    ind4 = u1 >= 0.75

    u1_scaled = np.where(ind1, 4 * u1,
                         np.where(ind2, 4 * (u1 - 0.25),
                                  np.where(ind3, 4 * (u1 - 0.5), 4 * (u1 - 0.75))))

    phi_h_base = phi_fun(u1_scaled, sat.nu[:, np.newaxis], sat.nv[:, np.newaxis])
    phi_h = np.where(ind1, phi_h_base,
                     np.where(ind2, phi_h_base + np.pi / 2,
                              np.where(ind3, phi_h_base + np.pi, phi_h_base + 3 / 2 * np.pi)))

    exp_val = sat.nu[:, np.newaxis] * np.cos(phi_h) ** 2 + sat.nv[:, np.newaxis] * np.sin(phi_h) ** 2 + 1
    theta_h = np.arccos(u2 ** (1 / exp_val))

    hx = np.sin(theta_h) * np.cos(phi_h)
    hy = np.sin(theta_h) * np.sin(phi_h)
    hz = np.cos(theta_h)

    # Integration
    SH = s_local[:, 0:1] * hx + s_local[:, 1:2] * hy + s_local[:, 2:3] * hz

    vx = 2 * SH * hx - s_local[:, 0:1]
    vy = 2 * SH * hy - s_local[:, 1:2]
    vz = 2 * SH * hz - s_local[:, 2:3]

    VH = vx * hx + vy * hy + vz * hz
    NV = vz
    NH = hz

    F = sat.F0[:, np.newaxis] + (1 - sat.F0[:, np.newaxis]) * (1 - VH) ** 5

    M = 1 / VH / np.maximum(NS[:, np.newaxis], NV)
    M = np.where(np.isinf(M), 0, M)

    # Weight
    W = np.abs(SH) * NV / NH * M
    tmp = W * F

    # For SRP
    mask = (NV > 0)
    cs_as_x = mask * tmp * vx
    cs_as_y = mask * tmp * vy
    cs_as_z = mask * tmp * vz

    # Specular component
    tmp_spec = np.column_stack([np.mean(cs_as_x, axis=1),
                                np.mean(cs_as_y, axis=1),
                                np.mean(cs_as_z, axis=1)])
    srp_cs = q_rotation(tmp_spec, q_inv(sat.qlb, scalar=4), scalar=4)

    # Shadowing
    sunlit_flag = (NS > 0).astype(float)

    tmp = sunlit_flag[:, np.newaxis] * (sun_b + srp_cd + srp_cs)
    sat.force = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * tmp
    sat.torque = np.cross(sat.pos, sat.force)

    # Output variables
    tmp_cd = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cd
    srp_cd_out = np.sum(tmp_cd, axis=0)

    tmp_cs = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cs
    srp_cs_out = np.sum(tmp_cs, axis=0)

    return sat, srp_cd_out, srp_cs_out


def srp_ct(sat: SatelliteModel, sun_b: np.ndarray, d: float, const,
           ndf: str = 'Beckmann', n_mc: int = 10000) -> tuple[SatelliteModel, np.ndarray, np.ndarray]:
    """
    # Numerically calculating SRP with Cook-Torrance model using importance sampling

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1 x 3
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        orbital constants
    ndf : str, optional
        NDF distribution function: 'Beckmann' (default) or 'Gauss'
    n_mc : int, optional
        number of random numbers for integration (default: 10^4)

    Returns
    -------
    sat : SatelliteModel
        updated satellite with force and torque at each facet
    srp_cd_out : np.ndarray
        total diffuse part of SRP, N, 1 x 3
    srp_cs_out : np.ndarray
        total specular part of SRP, N, 1 x 3

    References
    ----------
    Cook, R. L., & Torrance, K. E. (1982). A reflectance model for computer graphics.

    Revisions
    ---------
    y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_simple, read_sc, orbit_const
    """
    from ..conversion import km2au
    from ..attitude import q_rotation, q_inv

    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au ** 2

    sun_b = np.asarray(sun_b).flatten()
    sun_b = sun_b / np.linalg.norm(sun_b)

    # Diffuse (analytic)
    srp_cd = 2 / 3 * sat.Cd[:, np.newaxis] * sat.normal

    # Specular (numerical)
    n_facet = sat.normal.shape[0]

    # Transform to local frame
    sun_b_rep = np.tile(sun_b, (n_facet, 1))
    s_local = q_rotation(sun_b_rep, sat.qlb, scalar=4)

    NS = sat.normal @ sun_b

    # Monte-Carlo integration
    if ndf == 'Beckmann':
        u1 = np.random.rand(n_mc, 1)
        theta_h = np.arctan(np.sqrt(-sat.mCT[:, np.newaxis] ** 2 * np.log(u1.T)))
        phi_h = 2 * np.pi * np.random.rand(1, n_mc)

        hx = np.sin(theta_h) * np.cos(phi_h)
        hy = np.sin(theta_h) * np.sin(phi_h)
        hz = np.cos(theta_h)
    else:
        phi_r = 2 * np.pi * np.random.rand(1, n_mc)
        theta_r = np.pi / 2 * np.random.rand(1, n_mc)

        vx = np.sin(theta_r) * np.cos(phi_r)
        vy = np.sin(theta_r) * np.sin(phi_r)
        vz = np.cos(theta_r)

        hx = s_local[:, 0:1] + vx
        hy = s_local[:, 1:2] + vy
        hz = s_local[:, 2:3] + vz

        h_norm = np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
        hx = hx / h_norm
        hy = hy / h_norm
        hz = hz / h_norm

    # Integration
    SH = s_local[:, 0:1] * hx + s_local[:, 1:2] * hy + s_local[:, 2:3] * hz
    vx = 2 * SH * hx - s_local[:, 0:1]
    vy = 2 * SH * hy - s_local[:, 1:2]
    vz = 2 * SH * hz - s_local[:, 2:3]

    VH = vx * hx + vy * hy + vz * hz
    NV = vz
    NH = hz

    nest = (1 + np.sqrt(sat.F0[:, np.newaxis])) / (1 - np.sqrt(sat.F0[:, np.newaxis]))
    g = np.sqrt(nest ** 2 + VH ** 2 - 1)

    temp1 = 2 * NH * NV / VH
    temp2 = 2 * NH * NS[:, np.newaxis] / VH
    G = np.minimum(1, temp1)
    G = np.minimum(G, temp2)

    temp1 = (g - VH) ** 2 / 2 / (g + VH) ** 2
    temp2 = 1 + (VH * (g + VH) - 1) ** 2 / (VH * (g - VH) + 1) ** 2
    F = temp1 * temp2

    # Weight
    W = np.abs(SH) * G / NS[:, np.newaxis] / NH
    tmp = W * F
    tmp = np.where(np.isnan(tmp), 0, tmp)

    # SRP
    mask = (NV > 0)
    cs_ct_x = mask * tmp * vx
    cs_ct_y = mask * tmp * vy
    cs_ct_z = mask * tmp * vz

    # Specular component
    tmp_spec = np.column_stack([np.mean(cs_ct_x, axis=1),
                                np.mean(cs_ct_y, axis=1),
                                np.mean(cs_ct_z, axis=1)])
    srp_cs = q_rotation(tmp_spec, q_inv(sat.qlb, scalar=4), scalar=4)

    # Total SRP
    sunlit_flag = (NS > 0).astype(float)
    tmp = sun_b + srp_cd + srp_cs
    sat.force = sunlit_flag[:, np.newaxis] * coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * tmp
    sat.torque = np.cross(sat.pos, sat.force)

    # Output variables
    tmp_cd = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cd
    srp_cd_out = np.sum(tmp_cd, axis=0)

    tmp_cs = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cs
    srp_cs_out = np.sum(tmp_cs, axis=0)

    return sat, srp_cd_out, srp_cs_out


def srp_cannon(sat_am: float, sun_rel: np.ndarray, d: float, const,
               Cr: float = 1.2) -> np.ndarray:
    """
    # calculating solar radiation pressure (SRP) forces using cannonball model

    Parameters
    ----------
    sat_am : float
        area-to-mass ratio, m^2/kg
    sun_rel : np.ndarray
        sun vector from satellite to Sun expressed with inertial frame, nx3 unit vector
    d : float or np.ndarray
        distance between satellite and Sun, m
    const : OrbitalConstants
        orbital constants
    Cr : float, optional
        reflectivity coefficient (default: 1.2)

    Returns
    -------
    a_srp : np.ndarray
        SRP acceleration w.r.t. inertial frame, m/s^2, nx3 matrix

    Notes
    -----
    Simple cannonball model for SRP acceleration.

    References
    ----------
    Montenbruck, Oliver, & Eberhard Gill. Satellite Orbits. Springer Science & Business Media, 2012., p79

    Revisions
    ---------
    20221010  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    read_sc, orbit_const
    """
    from ..conversion import km2au

    sun_rel = np.asarray(sun_rel)
    d = np.asarray(d)

    # Convert distance to AU
    d_au = km2au(d / 1e3, const)  # AU

    coeff = -const.S0 / const.c * Cr * sat_am

    # Normalize sun vector
    if sun_rel.ndim == 1:
        sun_rel = sun_rel / np.linalg.norm(sun_rel)
        a_srp = coeff / d_au**2 * sun_rel
    else:
        sun_rel = sun_rel / np.linalg.norm(sun_rel, axis=1, keepdims=True)
        d_au = d_au.reshape(-1, 1)
        a_srp = coeff / d_au**2 * sun_rel

    return a_srp


def srp_as_uni(sat: 'SatelliteModel', sun_b: np.ndarray, d: float, const,
               n_mc: int = 100000) -> tuple['SatelliteModel', np.ndarray, np.ndarray]:
    """
    # Numerically calculating SRP with Ashikhmin-Shirley model (uniform sampling, not recommended)

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1x3 vector
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        orbital constants
    n_mc : int, optional
        number of sampling for Monte-Carlo integration (default: 10^5)

    Returns
    -------
    sat : SatelliteModel
        updated satellite with force and torque at each facet
    srp_cd_out : np.ndarray
        diffuse part of SRP, N, 1x3
    srp_cs_out : np.ndarray
        specular part of SRP, N, 1x3

    Notes
    -----
    SRP is written as:
    -S0/(c*r_AU^2) * A * (n^T s) * [s + integral(fr*(n^T v)*sin(theta_r)*v d_theta_r d_phi_r)]

    Uses uniform sampling over full sphere (not hemispherical).
    This method is not recommended - use srp_as with importance sampling instead.

    References
    ----------
    Ashikhmin, Michael, & Shirley, Peter. "An Anisotropic Phong BRDF Model."
    Journal of graphics tools, vol. 5, no. 2, 2000, pp. 25-32.

    Revisions
    ---------
    y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_as, srp_simple, read_sc, orbit_const
    """
    from ..conversion import km2au

    # Parameters
    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au**2

    sun_b = np.asarray(sun_b).flatten()
    sun_b = sun_b / np.linalg.norm(sun_b)

    NS = sat.normal @ sun_b  # n_facet x 1
    n_facet = sat.normal.shape[0]

    # Uniform sampling for spherical integration (not hemispherical)
    theta_r = np.pi * np.random.rand(n_mc)
    phi_r = 2 * np.pi * np.random.rand(n_mc)

    v = np.column_stack([np.sin(theta_r) * np.cos(phi_r),
                         np.sin(theta_r) * np.sin(phi_r),
                         np.cos(theta_r)])  # n_mc x 3

    h = sun_b + v  # bisector
    h = h / np.linalg.norm(h, axis=1, keepdims=True)  # n_mc x 3

    NH = sat.normal @ h.T  # n_facet x n_mc
    VH = np.sum(v * h, axis=1)  # n_mc
    NV = sat.normal @ v.T  # n_facet x n_mc

    # Diffuse (analytic)
    cd1 = 28 / 23 * sat.Cd / np.pi * (1 - sat.F0) * (1 - (1 - NS / 2)**5)  # n_facet
    srp_cd = np.zeros((n_facet, 3))
    srp_cd[:, 2] = np.sum(cd1) * 1573 / 2688 * np.pi

    # Specular (numerical)
    F = sat.F0[:, np.newaxis] + (1 - sat.F0[:, np.newaxis]) * (1 - VH)**5  # n_facet x n_mc
    k1 = np.sqrt((sat.nu + 1) * (sat.nv + 1)) / 8 / np.pi  # n_facet

    M = 1 / VH / np.maximum(NS[:, np.newaxis], NV)  # n_facet x n_mc
    M = np.where(np.isinf(M), 0, M)

    k2 = (sat.nu[:, np.newaxis] * (sat.uu @ h.T)**2 +
          sat.nv[:, np.newaxis] * (sat.uv @ h.T)**2) / (1 - NH**2 + 1e-10)
    D = NH ** k2  # n_facet x n_mc

    tmp = k1[:, np.newaxis] * F * M * D * NV * np.sin(theta_r)  # n_facet x n_mc

    mask = (NV > 0)
    cs_as_x = mask * tmp * v[:, 0]  # n_facet x n_mc
    cs_as_y = mask * tmp * v[:, 1]
    cs_as_z = mask * tmp * v[:, 2]

    pdf_as = 1 / np.pi * 1 / (2 * np.pi)  # probability

    srp_cs = np.column_stack([np.sum(cs_as_x, axis=1),
                               np.sum(cs_as_y, axis=1),
                               np.sum(cs_as_z, axis=1)])  # n_facet x 3
    srp_cs = srp_cs / pdf_as / n_mc

    # Shadowing
    sunlit_flag = (NS > 0).astype(float)

    tmp = sunlit_flag[:, np.newaxis] * (sun_b + srp_cd + srp_cs)
    sat.force = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * tmp
    sat.torque = np.cross(sat.pos, sat.force)

    # Output variables
    tmp_cd = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cd
    srp_cd_out = np.sum(tmp_cd, axis=0)

    tmp_cs = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cs
    srp_cs_out = np.sum(tmp_cs, axis=0)

    return sat, srp_cd_out, srp_cs_out


def srp_ct_uni(sat: 'SatelliteModel', sun_b: np.ndarray, d: float, const,
               ndf: str = 'Beckmann', n_mc: int = 1000000) -> tuple['SatelliteModel', np.ndarray, np.ndarray]:
    """
    # Numerically calculating SRP with Cook-Torrance model using uniform distribution (not recommended)

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1x3 vector
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        orbital constants
    ndf : str, optional
        NDF distribution function: 'Beckmann' (default) or 'Gauss'
    n_mc : int, optional
        number of random numbers for integration (default: 10^6)

    Returns
    -------
    sat : SatelliteModel
        updated satellite with force and torque at each facet
    srp_cd_out : np.ndarray
        diffuse part of SRP, N, 1x3
    srp_cs_out : np.ndarray
        specular part of SRP, N, 1x3

    Notes
    -----
    Uses uniform sampling over full sphere (not hemispherical).
    This method is not recommended - use srp_ct with importance sampling instead.

    References
    ----------
    Cook, R. L., & Torrance, K. E. (1982). A reflectance model for computer graphics.

    Revisions
    ---------
    20220201  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_ct, srp_simple, read_sc, orbit_const
    """
    from ..conversion import km2au

    # Parameters
    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au**2

    sun_b = np.asarray(sun_b).flatten()
    sun_b = sun_b / np.linalg.norm(sun_b)

    n_facet = sat.normal.shape[0]

    # Uniform sampling for spherical integration (not hemispherical)
    theta_r = np.pi * np.random.rand(n_mc)
    phi_r = 2 * np.pi * np.random.rand(n_mc)

    v = np.column_stack([np.sin(theta_r) * np.cos(phi_r),
                         np.sin(theta_r) * np.sin(phi_r),
                         np.cos(theta_r)])  # n_mc x 3

    h = np.tile(sun_b, (n_mc, 1)) + v
    h = h / np.linalg.norm(h, axis=1, keepdims=True)  # bisector, n_mc x 3

    theta_h = np.arccos(sat.normal @ h.T).T  # n_mc x n_facet -> transpose for broadcasting

    VH = np.sum(v * h, axis=1)  # n_mc
    NV = sat.normal @ v.T  # n_facet x n_mc
    NS = sat.normal @ sun_b  # n_facet
    NH = sat.normal @ h.T  # n_facet x n_mc

    # Diffuse (analytic)
    srp_cd = 2 / 3 * sat.Cd[:, np.newaxis] * sat.normal

    # Specular (numerical)
    theta_h_t = theta_h.T  # n_facet x n_mc for broadcasting

    if ndf == 'Beckmann':
        D = np.exp(-(np.tan(theta_h_t) / sat.mCT[:, np.newaxis])**2)  # Beckmann distribution
        D = D / np.pi / sat.mCT[:, np.newaxis]**2 / np.cos(theta_h_t)**4
    else:
        D = np.exp(-(theta_h_t / sat.mCT[:, np.newaxis])**2)  # Gaussian distribution

    nest = (1 + np.sqrt(sat.F0)) / (1 - np.sqrt(sat.F0))
    g = np.sqrt(nest[:, np.newaxis]**2 + VH**2 - 1)

    temp1 = 2 * NH * NV / VH
    temp2 = 2 * NH * NS[:, np.newaxis] / VH
    G = np.minimum(1, temp1)
    G = np.minimum(G, temp2)

    temp1 = (g - VH)**2 / 2 / (g + VH)**2
    temp2 = 1 + (VH * (g + VH) - 1)**2 / (VH * (g - VH) + 1)**2
    F = temp1 * temp2

    temp = D * G * F / NS[:, np.newaxis] / NV / 4
    temp = temp * NV * np.sin(theta_r)

    mask = (NV > 0)
    cs_ct_x = mask * temp * v[:, 0]
    cs_ct_y = mask * temp * v[:, 1]
    cs_ct_z = mask * temp * v[:, 2]

    # probability
    p_ct = 1 / np.pi * 1 / (2 * np.pi)

    srp_cs = np.column_stack([np.sum(cs_ct_x, axis=1),
                               np.sum(cs_ct_y, axis=1),
                               np.sum(cs_ct_z, axis=1)])
    srp_cs = srp_cs / p_ct / n_mc

    # Shadowing
    sunlit_flag = (NS > 0).astype(float)

    temp = sun_b + srp_cd + srp_cs
    sat.force = sunlit_flag[:, np.newaxis] * coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * temp
    sat.torque = np.cross(sat.pos, sat.force)

    # Output variables
    tmp_cd = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cd
    srp_cd_out = np.sum(tmp_cd, axis=0)

    tmp_cs = coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sunlit_flag[:, np.newaxis] * srp_cs
    srp_cs_out = np.sum(tmp_cs, axis=0)

    return sat, srp_cd_out, srp_cs_out


def srp_ct_interp(sat: 'SatelliteModel', sun_b: np.ndarray, d: float, const,
                  correction_para: dict) -> tuple['SatelliteModel', np.ndarray, np.ndarray]:
    """
    # SRP with Cook-Torrance model using correction parameters and interpolation

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1x3 vector
    d : float
        distance between satellite and Sun, m
    const : OrbitalConstants
        orbital constants
    correction_para : dict
        correction parameters with keys:
        - 'theta_i_span': incidence angle span
        - 'm_span': roughness parameter span
        - 'delta_s': correction for specular in sun direction
        - 'delta_n': correction for specular in normal direction

    Returns
    -------
    sat : SatelliteModel
        updated satellite with force and torque at each facet
    srp_cd_out : np.ndarray
        total diffuse part of SRP, N, 1x3 vector
    srp_cs_out : np.ndarray
        total specular part of SRP, N, 1x3 vector

    Notes
    -----
    Uses pre-computed correction parameters for fast SRP computation.

    References
    ----------
    NA

    Revisions
    ---------
    y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_ct, srp_simple, read_sc, orbit_const
    """
    from ..conversion import km2au
    from scipy.interpolate import RegularGridInterpolator

    # Parameters
    d_au = km2au(d / 1e3, const)  # AU
    S0 = const.S0  # Solar constant, W/m^2
    c = const.c  # light speed, m/s
    coeff = -S0 / c / d_au**2

    sun_b = np.asarray(sun_b).flatten()
    sun_b = sun_b / np.linalg.norm(sun_b)

    # Diffuse (analytic) and specular (corrected)
    NS = sat.normal @ sun_b  # n_facet
    theta_i = np.arccos(np.clip(NS, -1, 1))  # n_facet, clamped

    # Interpolation
    theta_i_span = correction_para['theta_i_span']
    m_span = correction_para['m_span']
    delta_s_data = correction_para['delta_s']
    delta_n_data = correction_para['delta_n']

    # Create interpolators
    interp_delta_s = RegularGridInterpolator(
        (m_span, theta_i_span), delta_s_data,
        method='linear', bounds_error=False, fill_value=1.0
    )
    interp_delta_n = RegularGridInterpolator(
        (m_span, theta_i_span), delta_n_data,
        method='linear', bounds_error=False, fill_value=1.0
    )

    # Query points (m, theta_i)
    points = np.column_stack([sat.mCT, theta_i])
    delta_s = interp_delta_s(points)
    delta_n = interp_delta_n(points)

    # Force calculation (per area unit, scaled by coeff)
    f_corrected = ((1 - delta_s * sat.Cs)[:, np.newaxis] * sun_b +
                   (2/3 * sat.Cd + 2 * NS * delta_n * sat.Cs)[:, np.newaxis] * sat.normal)

    # Total SRP
    sunlit_flag = (NS > 0).astype(float)
    sat.force = sunlit_flag[:, np.newaxis] * coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * f_corrected
    sat.torque = np.cross(sat.pos, sat.force)

    # Output variables
    # Total SRP force
    srp_total = np.sum(sat.force, axis=0)

    # Separation (Approximate)
    # Diffuse part (term with Cd)
    f_diffuse = sunlit_flag[:, np.newaxis] * coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * (2/3 * sat.Cd[:, np.newaxis] * sat.normal)
    srp_cd_out = np.sum(f_diffuse, axis=0)

    srp_impinged = sunlit_flag[:, np.newaxis] * coeff * sat.area[:, np.newaxis] * NS[:, np.newaxis] * sun_b
    srp_impinged_total = np.sum(srp_impinged, axis=0)

    # Specular part (rest)
    srp_cs_out = srp_total - srp_cd_out - srp_impinged_total

    return sat, srp_cd_out, srp_cs_out


def ct_m(sat: 'SatelliteModel', v: np.ndarray, sun_b: np.ndarray) -> np.ndarray:
    """
    # Calculating remaining term M in the Cook-Torrance model

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration read with read_sc
    v : np.ndarray
        reference vector, 1x3 vector
    sun_b : np.ndarray
        sun vector from satellite to Sun in body-fixed frame, 1x3 vector

    Returns
    -------
    M : np.ndarray
        remaining term M(v) = G(v) * F(v) / 4

    Notes
    -----
    Calculates the non-NDF terms in the Cook-Torrance model.

    References
    ----------
    Analytic Approximation of High-Fidelity Solar Radiation Pressure.

    Revisions
    ---------
    20200915  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_ct
    """
    v = np.asarray(v).flatten()
    sun_b = np.asarray(sun_b).flatten()

    n = sat.normal  # normal vectors, n x 3 matrix

    # bisector vector
    h = v + sun_b
    h = h / np.linalg.norm(h)

    # Dot products
    VH = np.dot(v, h)
    NH = n @ h
    NV = n @ v
    NS = n @ sun_b

    # Specular
    nest = (1 + np.sqrt(sat.F0)) / (1 - np.sqrt(sat.F0))
    g = np.sqrt(nest**2 + VH**2 - 1)

    temp1 = 2 * NH * NV / VH
    temp2 = 2 * NH * NS / VH

    G = np.minimum(1, temp1)
    G = np.minimum(G, temp2)

    temp1 = (g - VH)**2 / 2 / (g + VH)**2
    temp2 = 1 + (VH * (g + VH) - 1)**2 / (VH * (g - VH) + 1)**2

    F = temp1 * temp2

    M = np.sum(G * F) / 4

    return M


def make_coeff_table(m_span: np.ndarray = None, d_theta: float = 3.0,
                     F0: float = 0.5, n_mc: int = 1000000) -> dict:
    """
    # Make correction parameters for SRP with Cook-Torrance model

    Parameters
    ----------
    m_span : np.ndarray, optional
        roughness parameter span (default: [0.02, 0.05, 0.145, 0.275, 0.45])
    d_theta : float, optional
        theta increment in degrees (default: 3.0)
    F0 : float, optional
        Fresnel reflectance at normal incidence (default: 0.5)
    n_mc : int, optional
        number of Monte-Carlo samples (default: 10^6)

    Returns
    -------
    correction_para : dict
        correction parameters with keys:
        - 'delta_s': correction for specular in sun direction, shape (len(m_span), n_theta)
        - 'delta_n': correction for specular in normal direction, shape (len(m_span), n_theta)
        - 'm_span': roughness parameter span
        - 'theta_i_span': incidence angle span in radians

    Notes
    -----
    Generates correction parameters table for given F0 value.
    This is a computationally expensive function.

    References
    ----------
    NA

    Revisions
    ---------
    20230101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    srp_ct_interp
    """
    from scipy.special import expi
    from ..object import SatelliteModel
    from ..orbit import OrbitalConstants

    if m_span is None:
        m_span = np.array([0.02, 0.05, 0.145, 0.275, 0.45])

    theta_i_span = np.deg2rad(np.arange(0, 90 + d_theta, d_theta))
    n_theta = len(theta_i_span)
    n_m = len(m_span)

    # Configuration
    phi_i = np.deg2rad(-90)

    # Create simple satellite model for single facet
    sat = SatelliteModel()
    sat.pos = np.array([[0, 0, 0]])
    sat.normal = np.array([[0, 0, 1]])
    sat.area = np.array([1.0])
    sat.qlb = np.array([[0, 0, 0, 1]])
    sat.F0 = np.array([F0])
    sat.Cd = np.array([0.5])
    sat.Cs = np.array([F0])
    sat.Ca = np.array([0.0])
    sat.mCT = np.array([0.1])  # placeholder
    sat.nu = np.array([1.0])
    sat.nv = np.array([1.0])

    # Constants
    const = OrbitalConstants()

    # Distance from sat to sun (1 AU)
    d = const.AU * 1e3  # m
    d_au = 1.0
    S0 = const.S0
    c = const.c

    delta_s = np.zeros((n_m, n_theta))
    delta_n = np.zeros((n_m, n_theta))

    for i, m_val in enumerate(m_span):
        sat.mCT = np.array([m_val])

        for j, theta_i in enumerate(theta_i_span):
            # Sun direction
            sun_b = np.array([np.sin(theta_i) * np.cos(phi_i),
                             np.sin(theta_i) * np.sin(phi_i),
                             np.cos(theta_i)])

            NS = sat.normal[0] @ sun_b
            coeff = -S0 / c / d_au**2 * sat.area[0] * NS

            # Compute true SRP using CT model
            _, _, srp_cs = srp_ct(sat, sun_b, d, const, ndf='Beckmann', n_mc=n_mc)

            # Calculate correction parameters
            if abs(NS) > 1e-10 and abs(sat.Cs[0] * NS**2 - sat.Cs[0]) > 1e-10:
                delta_s[i, j] = (np.dot(srp_cs, sun_b) - np.dot(srp_cs, sat.normal[0]) * NS) / (sat.Cs[0] * NS**2 - sat.Cs[0])
                delta_s[i, j] = delta_s[i, j] / coeff

            if abs(NS) > 1e-10 and abs(2 * sat.Cs[0] * NS**3 - 2 * sat.Cs[0] * NS) > 1e-10:
                delta_n[i, j] = (np.dot(srp_cs, sun_b) * NS - np.dot(srp_cs, sat.normal[0])) / (2 * sat.Cs[0] * NS**3 - 2 * sat.Cs[0] * NS)
                delta_n[i, j] = delta_n[i, j] / coeff

            # Special case for near-normal incidence
            if theta_i < np.deg2rad(1):
                m_ct = m_val
                if m_ct < 0.045:
                    m_ct = 1 - 0.0013 * m_ct - 1.9634 * m_ct**2

                # Exponential integral approximation
                tmp1 = expi(-1/m_ct**2) + (4 + 4/m_ct**2) * expi(-2/m_ct**2) - (5 + 4/m_ct**2) * expi(-4/3/m_ct**2)
                tmp2 = (3 + 6/m_ct**2) * np.exp(-1/3/m_ct**2) - (2 + 4/m_ct**2) * np.exp(-1/m_ct**2) - 1
                delta_s[i, j] = 2 * np.exp(1/m_ct**2) / m_ct**2 * tmp1 + tmp2
                delta_n[i, j] = delta_s[i, j]

    correction_para = {
        'delta_s': delta_s,
        'delta_n': delta_n,
        'm_span': m_span,
        'theta_i_span': theta_i_span,
    }

    return correction_para


# %[appendix]{"version":"1.0"}
