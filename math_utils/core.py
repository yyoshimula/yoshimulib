"""
Core math utility functions
Python conversion from yMATLAB/math/
"""

import numpy as np


def skew(x: np.ndarray) -> np.ndarray:
    """
    # make skew-symmetric matrix from vector x

    Parameters
    ----------
    x : np.ndarray
        3-element vector

    Returns
    -------
    S : np.ndarray
        3x3 skew-symmetric matrix

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
    NA
    """
    S = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])

    return S


# %[appendix]{"version":"1.0"}


def wrap_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """
    # Wrap angle in radians to the interval [-pi, pi)

    Parameters
    ----------
    angle : float or np.ndarray
        angle in radians

    Returns
    -------
    wrapped : float or np.ndarray
        angle wrapped to [-pi, pi)

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
    np.mod
    """
    wrapped = np.mod(angle + np.pi, 2 * np.pi) - np.pi

    return wrapped


# %[appendix]{"version":"1.0"}


def norm_row(A: np.ndarray) -> np.ndarray:
    """
    # Normalize row vectors of matrix

    Parameters
    ----------
    A : np.ndarray
        Input matrix (n x m)

    Returns
    -------
    B : np.ndarray
        Matrix with normalized rows

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
    np.linalg.norm
    """
    Anorm = np.linalg.norm(A, axis=-1, keepdims=True)

    # Handle case where Anorm might be 0 or very small
    mask = Anorm > np.finfo(float).eps
    B = np.where(mask, A / Anorm, A)

    return B


# %[appendix]{"version":"1.0"}
