"""
Spherical Gaussian functions
Python conversion from yMATLAB/sphericalGaussian/
"""

import numpy as np


def sg(x: np.ndarray, y: np.ndarray, z: np.ndarray,
       p: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """
    # Spherical Gaussian

    Parameters
    ----------
    x : np.ndarray
        x-position on a sphere, N x M matrix
    y : np.ndarray
        y-position on a sphere, N x M matrix
    z : np.ndarray
        z-position on a sphere, N x M matrix
    p : np.ndarray
        lobe directional vector, 1 x 3 vector
    lam : float
        sharpness, scalar
    mu : float
        amplitude, scalar

    Returns
    -------
    G : np.ndarray
        spherical Gaussian, same shape as input

    Notes
    -----
    Spherical Gaussian is defined as:
    G(x, p, lambda, mu) = mu * exp(lambda * (x^T * p - 1))

    References
    ----------
    NA

    Revisions
    ---------
    20200911  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sg_mix, asg
    """
    p = np.asarray(p).flatten()
    p = p / np.linalg.norm(p)

    G = mu * np.exp(lam * (x * p[0] + y * p[1] + z * p[2] - 1))

    return G


# %[appendix]{"version":"1.0"}


def asg(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        lam: float, mu: float, c: float) -> np.ndarray:
    """
    # Anisotropic Spherical Gaussian

    Parameters
    ----------
    vx : np.ndarray
        x-position on sphere, N x M matrix
    vy : np.ndarray
        y-position on sphere, N x M matrix
    vz : np.ndarray
        z-position on sphere, N x M matrix
    x : np.ndarray
        tangent axis, 1 x 3 vector
    y : np.ndarray
        bi-tangent axis, 1 x 3 vector
    z : np.ndarray
        lobe axis, 1 x 3 vector
    lam : float
        bandwidth along x-axis, scalar
    mu : float
        bandwidth along y-axis, scalar
    c : float
        amplitude, scalar

    Returns
    -------
    aSG : np.ndarray
        anisotropic spherical Gaussian, same shape as input

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20231004  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sg_mix, sg
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    VX = vx * x[0] + vy * x[1] + vz * x[2]
    VY = vx * y[0] + vy * y[1] + vz * y[2]
    VZ = vx * z[0] + vy * z[1] + vz * z[2]

    S = np.maximum(0, VZ)
    aSG = c * S * np.exp(-lam * VX ** 2 - mu * VY ** 2)

    return aSG


# %[appendix]{"version":"1.0"}


def sg_mix(p1: np.ndarray, lam1: float | np.ndarray, mu1: float | np.ndarray,
           p2: np.ndarray, lam2: float | np.ndarray, mu2: float | np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Spherical Gaussian mixture (SG multiplication)

    Parameters
    ----------
    p1 : np.ndarray
        lobe axis 1, n x 3 vector
    lam1 : float or np.ndarray
        sharpness 1, scalar or n x 1
    mu1 : float or np.ndarray
        amplitude 1, scalar or n x 1
    p2 : np.ndarray
        lobe axis 2, n x 3 vector
    lam2 : float or np.ndarray
        sharpness 2, scalar or n x 1
    mu2 : float or np.ndarray
        amplitude 2, scalar or n x 1

    Returns
    -------
    p3 : np.ndarray
        mixed lobe axis, n x 3 vector
    lam3 : np.ndarray
        mixed sharpness, n x 1
    mu3 : np.ndarray
        mixed amplitude, n x 1

    Notes
    -----
    Spherical Gaussian:
    G(x, p, lambda, mu) = mu * exp(lambda * (x^T * p - 1))

    Spherical Gaussian mixture:
    G(x; p1, lam1, mu1) * G(x; p2, lam2, mu2) = G(x; p_m, lam_m, mu_m)

    where:
    p_m = lam1 * p1 + lam2 * p2
    lam_m = ||p_m||
    p_m_bar = p_m / ||p_m||
    mu_m = mu1 * mu2 * exp(lam_m - (lam1 + lam2))

    References
    ----------
    NA

    Revisions
    ---------
    20210206  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sg
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)

    lam1 = np.atleast_1d(lam1).reshape(-1, 1)
    lam2 = np.atleast_1d(lam2).reshape(-1, 1)
    mu1 = np.atleast_1d(mu1).reshape(-1, 1)
    mu2 = np.atleast_1d(mu2).reshape(-1, 1)

    # Sharpness
    p_m = lam1 * p1 + lam2 * p2
    lam3 = np.linalg.norm(p_m, axis=1, keepdims=True)

    # Lobe axis
    p3 = p_m / lam3

    # Amplitude
    mu3 = mu1 * mu2 * np.exp(lam3 - (lam1 + lam2))

    return p3, lam3.flatten(), mu3.flatten()


# %[appendix]{"version":"1.0"}
