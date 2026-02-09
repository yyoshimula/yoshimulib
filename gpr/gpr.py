"""
Gaussian Process Regression functions
Python conversion from yMATLAB/gpr/
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def kernel_gauss(x: np.ndarray, xp: np.ndarray, hyp_para: np.ndarray) -> float:
    """
    # Gaussian kernel for each input vector

    Parameters
    ----------
    x : np.ndarray
        x to be calculated, 1xn or nx1 vector
    xp : np.ndarray
        x' to be calculated, 1xn or nx1 vector
    hyp_para : np.ndarray
        hyper parameters [theta_1, theta_2]

    Returns
    -------
    k : float
        kernel value

    Notes
    -----
    k(x_i, x_j; theta_1, theta_2) := theta_1 * exp(-||x_i - x_j||^2 / theta_2)

    References
    ----------
    NA

    Revisions
    ---------
    20230614  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gpr_cov, kernel_gauss_mat
    """
    x = np.asarray(x).flatten()
    xp = np.asarray(xp).flatten()

    k = hyp_para[0] * np.exp(-np.dot(x - xp, x - xp) / hyp_para[1])

    return k


# %[appendix]{"version":"1.0"}


def kernel_gauss_mat(x_train: np.ndarray, hyp_para: np.ndarray, sig_n: float) -> np.ndarray:
    """
    # Gaussian kernel matrix for each training data

    Parameters
    ----------
    x_train : np.ndarray
        training data, d x n matrix (d samples, n features)
    hyp_para : np.ndarray
        hyper parameters for kernel [theta_1, theta_2]
    sig_n : float
        standard deviation of additive noise

    Returns
    -------
    K : np.ndarray
        Gaussian kernel matrix, d x d matrix

    Notes
    -----
    Uses scipy.spatial.distance.pdist for efficient computation.

    References
    ----------
    NA

    Revisions
    ---------
    20230614  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gpr_cov, kernel_gauss
    """
    d = x_train.shape[0]

    # Compute kernel matrix using pdist
    K = hyp_para[0] * np.exp(-squareform(pdist(x_train)) ** 2 / hyp_para[1])

    # Add noise term
    K = K + sig_n ** 2 * np.eye(d)

    return K


# %[appendix]{"version":"1.0"}


def gpr_mean(x_ast: np.ndarray, x_train: np.ndarray, x_mean: np.ndarray,
             y_train: np.ndarray, y_mean: np.ndarray, L: np.ndarray,
             hyp_para: np.ndarray) -> tuple[np.ndarray, float]:
    """
    # mean value of Gaussian Process Regression

    Parameters
    ----------
    x_ast : np.ndarray
        test data, m x n matrix
    x_train : np.ndarray
        training data, d x n matrix
    x_mean : np.ndarray
        mean of training data
    y_train : np.ndarray
        training output data, d x 1 vector
    y_mean : np.ndarray
        mean of training output
    L : np.ndarray
        Cholesky decomposition of kernel matrix (lower triangular)
    hyp_para : np.ndarray
        hyper parameters for kernel

    Returns
    -------
    y_pred : np.ndarray
        predicted mean value for test data, m x 1 matrix
    log_p : float
        log marginal likelihood, log(p(y|X))

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20230614  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gpr_cov
    """
    d = x_train.shape[0]  # number of training data
    m = x_ast.shape[0]  # number of test data

    # Calculate k_*
    k_ast = np.zeros((d, m))

    for i in range(m):
        for j in range(d):
            k_ast[j, i] = kernel_gauss(x_ast[i, :], x_train[j, :], hyp_para)

    # Use Cholesky decomposition (numerically stable and fast)
    y_diff = y_train - x_mean
    alp = np.linalg.solve(L.T, np.linalg.solve(L, y_diff))

    y_pred = x_mean + k_ast.T @ alp

    # Log marginal likelihood
    log_p = -0.5 * y_train.T @ alp - np.sum(np.log(np.diag(L))) - d / 2 * np.log(2 * np.pi)

    return y_pred, float(log_p)


# %[appendix]{"version":"1.0"}


def gpr_cov(x_ast: np.ndarray, x_train: np.ndarray,
            K_inv: np.ndarray, hyp_para: np.ndarray) -> np.ndarray:
    """
    # covariance matrix of Gaussian Process Regression

    Parameters
    ----------
    x_ast : np.ndarray
        test data, m x n matrix
    x_train : np.ndarray
        training data, d x n matrix
    K_inv : np.ndarray
        inverse matrix of the kernel matrix, d x d matrix
    hyp_para : np.ndarray
        hyper parameters for kernel

    Returns
    -------
    sig_pred : np.ndarray
        predicted variance for test data, m x 1 matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20230614  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    gpr_mean
    """
    d = x_train.shape[0]  # number of training data
    m = x_ast.shape[0]  # number of test data

    # k_{**}
    k_ast_ast = np.zeros(m)

    # k_*
    k_ast = np.zeros((d, m))

    for i in range(m):
        k_ast_ast[i] = kernel_gauss(x_ast[i, :], x_ast[i, :], hyp_para)

        for j in range(d):
            k_ast[j, i] = kernel_gauss(x_ast[i, :], x_train[j, :], hyp_para)

    sig_pred = k_ast_ast - np.diag(k_ast.T @ K_inv @ k_ast)

    return sig_pred


# %[appendix]{"version":"1.0"}
