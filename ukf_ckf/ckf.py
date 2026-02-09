"""
Cubature Kalman Filter functions
Python conversion from yMATLAB/ukfCkf/
"""

import numpy as np
from typing import Callable


def ckf_sigma(P: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    # calculating sigma cubature points

    Parameters
    ----------
    P : np.ndarray
        covariance matrix, n x n matrix
    x : np.ndarray
        state vector, n x 1 or 1 x n vector

    Returns
    -------
    X_out : np.ndarray
        cubature sigma points: 2n x n matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20250617  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    ckf_cov
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # Square root matrix
    S = np.linalg.cholesky(P)  # P = S @ S.T

    # Cubature sigma points
    xi = np.sqrt(n) * S  # n x n matrix
    X_out = np.vstack([x + xi.T, x - xi.T])  # 2n x n matrix

    return X_out


# %[appendix]{"version":"1.0"}


def ckf_cov(x_est: np.ndarray, X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    # covariance using sigma cubature points for CKF

    Parameters
    ----------
    x_est : np.ndarray
        state vector: 1 x n vector
    X : np.ndarray
        sigma cubature points: 2n x n matrix
    Q : np.ndarray
        process noise matrix, n x n matrix

    Returns
    -------
    P_out : np.ndarray
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
    ckf_sigma
    """
    n = X.shape[1]
    P = np.zeros((n, n))

    # Covariance sum
    for i in range(2 * n):
        P += np.outer(X[i, :], X[i, :])
    P = P / (2 * n)

    x_est = np.asarray(x_est).flatten()
    P_out = P - np.outer(x_est, x_est) + Q

    return P_out


# %[appendix]{"version":"1.0"}


def ckf_corr_gain(x_est: np.ndarray, X: np.ndarray, y_est: np.ndarray,
                  Y: np.ndarray, R: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Correlated covariances and Kalman gain for cubature Kalman filter (CKF)

    Parameters
    ----------
    x_est : np.ndarray
        state vector: n x 1 vector
    X : np.ndarray
        sigma cubature points: 2n x n matrix
    y_est : np.ndarray
        a priori estimated measurement vector: m x 1 vector
    Y : np.ndarray
        measurement sigma cubature points: 2n x m matrix
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
    ckf_cov
    """
    x_est = np.asarray(x_est).flatten()
    y_est = np.asarray(y_est).flatten()
    n = len(x_est)
    m = len(y_est)

    Pyy = np.zeros((m, m))
    Pxy = np.zeros((n, m))

    # Covariance
    for i in range(2 * n):
        Pyy += np.outer(Y[i, :], Y[i, :])
    Pyy = Pyy / (2 * n) - np.outer(y_est, y_est) + R

    # Cross correlation
    for i in range(2 * n):
        Pxy += np.outer(X[i, :], Y[i, :])
    Pxy = Pxy / (2 * n) - np.outer(x_est, y_est)

    # Kalman gain
    K = Pxy @ np.linalg.inv(Pyy)

    return Pyy, Pxy, K


# %[appendix]{"version":"1.0"}


def srckf_sigma(P: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    # calculating sigma cubature points (spherical radial)

    Parameters
    ----------
    P : np.ndarray
        covariance matrix, n x n matrix
    x : np.ndarray
        state vector, n x 1 or 1 x n vector

    Returns
    -------
    X_out : np.ndarray
        cubature sigma points: (2n+1) x n matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20250617  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    ckf_cov
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # Cholesky decomposition
    S = np.linalg.cholesky(P)

    # Spherical sigma points
    xi = np.sqrt(n) * S  # n x n matrix

    # Sigma point construction
    X_out = np.vstack([x, x + xi.T, x - xi.T])  # (2n+1) x n

    return X_out


# %[appendix]{"version":"1.0"}


def srckf_cov(x_est: np.ndarray, X: np.ndarray, wc: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    # covariance using (spherical radial) sigma cubature points

    Parameters
    ----------
    x_est : np.ndarray
        state vector: 1 x n vector
    X : np.ndarray
        sigma cubature points: (2n+1) x n matrix
    wc : np.ndarray
        weight for covariance, (2n+1,)
    Q : np.ndarray
        process noise matrix, n x n matrix

    Returns
    -------
    P_out : np.ndarray
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
    srckf_sigma
    """
    n_sigma = X.shape[0]
    n = len(x_est)
    P_out = np.zeros((n, n))

    x_est = np.asarray(x_est).flatten()

    for i in range(n_sigma):
        dx = X[i, :] - x_est
        P_out += wc[i] * np.outer(dx, dx)

    P_out += Q

    return P_out


# %[appendix]{"version":"1.0"}


def srckf_corr_gain(x_est: np.ndarray, X: np.ndarray, y_est: np.ndarray,
                    Y: np.ndarray, wc: np.ndarray, R: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Correlated covariances and Kalman gain for (spherical radial) cubature Kalman filter

    Parameters
    ----------
    x_est : np.ndarray
        state vector: n x 1 vector
    X : np.ndarray
        sigma cubature points: (2n+1) x n matrix
    y_est : np.ndarray
        a priori estimated measurement vector: m x 1 vector
    Y : np.ndarray
        measurement sigma cubature points: (2n+1) x m matrix
    wc : np.ndarray
        weight for covariance: (2n+1,) vector
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
    srckf_cov
    """
    n_sigma = X.shape[0]
    x_est = np.asarray(x_est).flatten()
    y_est = np.asarray(y_est).flatten()
    n = len(x_est)
    m = len(y_est)

    Pyy = np.zeros((m, m))
    Pxy = np.zeros((n, m))

    for i in range(n_sigma):
        dy = Y[i, :] - y_est
        dx = X[i, :] - x_est
        Pyy += wc[i] * np.outer(dy, dy)
        Pxy += wc[i] * np.outer(dx, dy)

    Pyy += R
    K = Pxy @ np.linalg.inv(Pyy)

    return Pyy, Pxy, K


# %[appendix]{"version":"1.0"}
