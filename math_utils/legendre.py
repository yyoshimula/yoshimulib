"""
Associated Legendre polynomials
Python conversion from yMATLAB/math/
"""

import numpy as np


def _double_factorial(n: int) -> int:
    """
    # Compute double factorial n!!

    Parameters
    ----------
    n : int
        Input integer

    Returns
    -------
    result : int
        n!! value

    Notes
    -----
    n!! = n * (n-2) * (n-4) * ... * (2 or 1)
    """
    if n <= 0:
        return 1
    elif n % 2 == 0:
        # Even case: n!! = n * (n-2) * (n-4) * ... * 2
        return np.prod(np.arange(2, n + 1, 2))
    else:
        # Odd case: n!! = n * (n-2) * (n-4) * ... * 1
        return np.prod(np.arange(1, n + 1, 2))


def _legendre_recursive(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """
    # Compute Associated Legendre polynomial P_l^m(x) using recursion

    Parameters
    ----------
    l : int
        Degree (non-negative integer)
    m : int
        Order (0 <= m <= l)
    x : np.ndarray
        Evaluation points

    Returns
    -------
    P : np.ndarray
        P_l^m(x) values
    """
    x = np.asarray(x)

    # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    if l == m:
        P = ((-1) ** m) * _double_factorial(2 * m - 1) * ((1 - x ** 2) ** (m / 2))
        return P

    # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    if l == m + 1:
        Pmm = ((-1) ** m) * _double_factorial(2 * m - 1) * ((1 - x ** 2) ** (m / 2))
        P = x * (2 * m + 1) * Pmm
        return P

    # Use recurrence relation
    # (l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m
    Pmm = ((-1) ** m) * _double_factorial(2 * m - 1) * ((1 - x ** 2) ** (m / 2))
    Pmp1m = x * (2 * m + 1) * Pmm

    P = np.zeros_like(x)
    for k in range(m + 2, l + 1):
        P = (x * (2 * k - 1) * Pmp1m - (k + m - 1) * Pmm) / (k - m)
        Pmm = Pmp1m
        Pmp1m = P

    return P


def associated_legendre(l: int, x: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    # Associated Legendre polynomial P_l^m(x)
    Compute for all orders m = 0, 1, ..., l

    Parameters
    ----------
    l : int
        Degree (non-negative integer)
    x : float or np.ndarray
        Evaluation points (scalar or vector)

    Returns
    -------
    P : np.ndarray
        (l+1) x length(x) matrix, P[i, j] = P_l^{m_i}(x_j)
    m_values : np.ndarray
        Order vector [0, 1, ..., l]

    Notes
    -----
    Uses recursive relation for computation.

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    scipy.special.lpmv
    """
    x = np.atleast_1d(np.asarray(x))
    m_values = np.arange(l + 1)
    P = np.zeros((len(m_values), len(x)))

    for idx, m in enumerate(m_values):
        P[idx, :] = _legendre_recursive(l, m, x)

    return P, m_values


# %[appendix]{"version":"1.0"}
