"""
Unscented Kalman Filter functions
Python conversion from yMATLAB/ukfCkf/
"""

import numpy as np
from typing import Callable


def ukf_init_para(n: int, ukf: dict = None) -> dict:
    """
    # Initialize UKF parameters and weights

    Parameters
    ----------
    n : int
        number of state variables
    ukf : dict, optional
        UKF parameters set in advance. The entries 'alp', 'bet', 'kappa',
        'lam', 'wm' and 'wc' are used as they are when given, and the
        missing ones are filled with the default values.

    Returns
    -------
    ukf : dict
        UKF parameter dictionary containing:
        - n: number of state variables
        - alp, bet, kappa: tuning parameters (default: 1e-4, 2, 3 - n)
        - lam: lambda parameter (default: alp^2 * (n + kappa) - n)
        - wm: weights for mean (2n+1,)
        - wc: weights for covariance (2n+1,)

    Notes
    -----
    MATLAB: ukfInitPara (formerly setUKFpara). The MATLAB field `lambda`
    corresponds to 'lam' and `n_` to 'n'.

    Revisions
    ---------
    20260302  y.yoshimura, renamed from setUKFpara, number of state variables n added

    See also
    --------
    ukf_sigma, ukf
    """
    ukf = dict(ukf) if ukf is not None else {}

    ukf['n'] = n

    # Tuning parameters
    ukf.setdefault('alp', 1e-4)
    ukf.setdefault('bet', 2)
    ukf.setdefault('kappa', 3 - n)
    ukf.setdefault('lam', ukf['alp'] ** 2 * (n + ukf['kappa']) - n)

    # Weights
    lam = ukf['lam']

    if 'wm' not in ukf:
        wm = np.zeros(2 * n + 1)
        wm[0] = lam / (n + lam)  # for mean
        wm[1:] = 1 / (2 * (n + lam))
        ukf['wm'] = wm
    ukf['wm'] = np.asarray(ukf['wm'], dtype=float).flatten()

    if 'wc' not in ukf:
        wc = np.zeros(2 * n + 1)
        wc[0] = lam / (n + lam) + 1 - ukf['alp'] ** 2 + ukf['bet']
        wc[1:] = ukf['wm'][1:]
        ukf['wc'] = wc
    ukf['wc'] = np.asarray(ukf['wc'], dtype=float).flatten()

    return ukf


# %[appendix]{"version":"1.0"}


def set_ukf_para(n: int, alp: float = None, bet: float = None, kappa: float = None) -> dict:
    """
    # Set UKF parameters and weights (former name of ukf_init_para)

    Parameters
    ----------
    n : int
        state dimension
    alp : float, optional
        tuning parameter (default: 1e-4)
    bet : float, optional
        tuning parameter (default: 2)
    kappa : float, optional
        tuning parameter (default: 3 - n)

    Returns
    -------
    ukf : dict
        UKF parameter dictionary containing:
        - n: number of state variables
        - alp, bet, kappa: tuning parameters
        - lam: lambda parameter
        - wm: weights for mean (2n+1,)
        - wc: weights for covariance (2n+1,)

    Notes
    -----
    Kept for backward compatibility; MATLAB renamed setUKFpara to ukfInitPara.

    Revisions
    ---------
    NA

    See also
    --------
    ukf_init_para, ukf_sigma
    """
    given = {'alp': alp, 'bet': bet, 'kappa': kappa}

    return ukf_init_para(n, {k: val for k, val in given.items() if val is not None})


# %[appendix]{"version":"1.0"}


def ukf_sigma(lam: float, P: np.ndarray, x: np.ndarray, method: str = 'svd') -> np.ndarray:
    """
    # calculating sigma points

    Parameters
    ----------
    lam : float
        tuning parameter of UKF
    P : np.ndarray
        covariance matrix, n x n matrix
    x : np.ndarray
        state vector, n x 1 or 1 x n vector
    method : str, optional
        'svd' (default) or 'chol' for matrix square root

    Returns
    -------
    X : np.ndarray
        sigma points: (2n+1) x n matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20240809 arguments modified
    20181210  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    ukf_cov
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # Square root matrix
    if method == 'svd':
        U, S, _ = np.linalg.svd(P)
        Psq = U @ np.diag(np.sqrt(S))
    elif method == 'chol':
        Psq = np.linalg.cholesky(P)
    else:
        raise ValueError("method must be 'svd' or 'chol'")

    # Sigma points
    sig = np.sqrt(n + lam) * Psq  # n x n matrix
    X = np.vstack([x, x + sig.T, x - sig.T])  # (2n+1) x n matrix

    return X


# %[appendix]{"version":"1.0"}


def ukf_cov(x_est: np.ndarray, X: np.ndarray, wc: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    # covariance using sigma points

    Parameters
    ----------
    x_est : np.ndarray
        state vector: 1 x n vector
    X : np.ndarray
        sigma points: (2n+1) x n matrix
    wc : np.ndarray
        weight for covariance, (2n+1,)
    Q : np.ndarray
        process noise matrix, n x n matrix

    Returns
    -------
    P_cov : np.ndarray
        a priori covariance: n x n matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20210209  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    ukf_sigma, ukf
    """
    x_est = np.asarray(x_est).flatten()
    n = len(x_est)
    P_cov = np.zeros((n, n))

    # Covariance sum
    for i in range(2 * n + 1):
        dx = X[i, :] - x_est
        P_cov += wc[i] * np.outer(dx, dx)

    P_cov += Q

    return P_cov


# %[appendix]{"version":"1.0"}


def ukf_corr_gain(x_est: np.ndarray, X: np.ndarray, y_est: np.ndarray,
                  Y: np.ndarray, wc: np.ndarray, R: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Correlated covariances and Kalman gain

    Parameters
    ----------
    x_est : np.ndarray
        state vector: n x 1 vector
    X : np.ndarray
        sigma points: (2n+1) x n matrix
    y_est : np.ndarray
        a priori estimated measurement vector: m x 1 vector
    Y : np.ndarray
        measurement sigma points: (2n+1) x m matrix
    wc : np.ndarray
        weight for covariance, (2n+1,)
    R : np.ndarray
        measurement noise matrix, m x m

    Returns
    -------
    Pyy : np.ndarray
        measurement covariance: m x m matrix
    Pxy : np.ndarray
        correlated covariance: n x m matrix
    K : np.ndarray
        Kalman gain: n x m matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20210209  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    ukf_cov
    """
    x_est = np.asarray(x_est).flatten()
    y_est = np.asarray(y_est).flatten()
    n = len(x_est)
    m = len(y_est)

    Pyy = np.zeros((m, m))
    Pxy = np.zeros((n, m))

    # Covariance
    for i in range(2 * n + 1):
        dy = Y[i, :] - y_est
        Pyy += wc[i] * np.outer(dy, dy)
    Pyy += R

    # Cross correlation
    for i in range(2 * n + 1):
        dx = X[i, :] - x_est
        dy = Y[i, :] - y_est
        Pxy += wc[i] * np.outer(dx, dy)

    # Kalman gain
    K = Pxy @ np.linalg.inv(Pyy)

    return Pyy, Pxy, K


# %[appendix]{"version":"1.0"}


def ukf(f: Callable, h: Callable, Q: np.ndarray, R: np.ndarray,
        lam: float, P: np.ndarray, x: np.ndarray, y: np.ndarray,
        wm: np.ndarray, wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    # Unscented Kalman Filter

    Parameters
    ----------
    f : Callable
        system discrete dynamics: x(k+1) = f(x(k))
    h : Callable
        observation function: y(k) = h(x(k))
    Q : np.ndarray
        system noise: n x n matrix
    R : np.ndarray
        observation noise: m x m matrix
    lam : float
        tuning parameter for UKF
    P : np.ndarray
        covariance matrix, n x n matrix
    x : np.ndarray
        state vector, n x 1 vector
    y : np.ndarray
        observation vector, m x 1 vector
    wm : np.ndarray
        weights for mean, (2n+1,)
    wc : np.ndarray
        weights for covariance, (2n+1,)

    Returns
    -------
    P_est : np.ndarray
        estimated covariance matrix, n x n matrix
    x_est : np.ndarray
        estimated state, n x 1 vector

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20181210  y.yoshimura

    See also
    --------
    ukf_sigma, ukf_cov, ukf_corr_gain
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Sigma points
    X = ukf_sigma(lam, P, x)

    # Sigma point propagation
    X_prop = np.array([f(xi) for xi in X])

    # A priori estimation
    x_est = np.sum(wm[:, np.newaxis] * X_prop, axis=0)

    # Covariance
    P = ukf_cov(x_est, X_prop, wc, Q)

    # Output sigma points
    Y = np.array([h(xi) for xi in X_prop])

    # A priori output estimation
    y_est = np.sum(wm[:, np.newaxis] * Y, axis=0)

    # Calculate correlation
    Pyy, Pxy, K = ukf_corr_gain(x_est, X_prop, y_est, Y, wc, R)

    # Gain and Update
    P_est = P - K @ Pyy @ K.T
    x_est = x_est + K @ (y - y_est)

    return P_est, x_est


# %[appendix]{"version":"1.0"}
