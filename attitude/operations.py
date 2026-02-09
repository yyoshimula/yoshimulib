"""
Advanced attitude operations
Python conversion from yMATLAB/attitude/
"""

import numpy as np
from .quaternion import q_mult, q_inv


def slerp(t: np.ndarray, scalar: int, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    # spherical linear interpolation (slerp)
    quaternionの線形補間

    Parameters
    ----------
    t : np.ndarray
        normalized time, i.e., 0 <= t <= 1
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    q1 : np.ndarray
        quaternions, 1x4 vector
    q2 : np.ndarray
        quaternions, 1x4 vector

    Returns
    -------
    qt : np.ndarray
        interpolated quaternions, nx4 matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sclerp
    """
    assert scalar in [0, 4], "quaternion definition is unclear"

    t = np.atleast_1d(t).flatten()  # column vector

    # Python: Ensure q1, q2 are 2D for q_mult/q_inv
    q1 = np.atleast_2d(q1)
    q2 = np.atleast_2d(q2)

    qtmp = q_mult(scalar, 0, q_inv(scalar, q1), q2)
    qtmp = qtmp.flatten()

    if scalar == 0:
        eTheta = np.arccos(qtmp[0])
        eTheta = eTheta * 2
        eAxis = qtmp[1:4]

    elif scalar == 4:
        eTheta = np.arccos(qtmp[3])  # Euler angle of rotation
        eTheta = eTheta * 2
        eAxis = qtmp[0:3]

    else:
        raise ValueError('quaternion definition is unclear')

    eAxis = eAxis / np.sin(eTheta / 2)  # Euler axis of rotation

    # Python: kron(ones(length(t),1), q1) -> np.tile(q1, (len(t), 1))
    q1tmp = np.tile(q1.flatten(), (len(t), 1))  # Nx4

    if scalar == 0:
        q_delta = np.column_stack([
            np.cos(t * eTheta / 2),
            eAxis * np.sin(t * eTheta / 2).reshape(-1, 1)
        ])
        qt = q_mult(scalar, 0, q1tmp, q_delta)

    elif scalar == 4:
        q_delta = np.column_stack([
            eAxis * np.sin(t * eTheta / 2).reshape(-1, 1),
            np.cos(t * eTheta / 2)
        ])
        qt = q_mult(scalar, 0, q1tmp, q_delta)

    else:
        raise ValueError('quaternion definition is unclear')

    return qt


# %[appendix]{"version":"1.0"}


def q_ave(q: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    """
    # quaternion averaging with scalar weights
    the definition of output quaternion is consistent with the input quaternion
    入力のquaternionと同じ定義のquaternionが出力される

    Parameters
    ----------
    q : np.ndarray
        quaternions, nx4 matrix
        q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        or
        q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    w : np.ndarray, optional
        scalar weights, nx1. Default is uniform weights (1/n for each)

    Returns
    -------
    qAveraged : np.ndarray
        averaged quaternion, 1x4 vector

    Notes
    -----
    NA

    References
    ----------
    Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y. (2007).
    Averaging Quaternions. Journal of Guidance, Control, and Dynamics,
    30(4), 1193-1197. https://doi.org/10.2514/1.28949

    Revisions
    ---------
    20230614  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2zyx
    """
    # arguments
    #     q (:,4) {mustBeNumeric}
    #     w (:,1) {mustBeNumeric} = ones(size(q,1), 1) ./ size(q,1)
    # end

    # Python: Ensure q is 2D
    q = np.atleast_2d(q)
    n = q.shape[0]

    if w is None:
        w = np.ones(n) / n
    else:
        w = np.atleast_1d(w).flatten()

    M = np.zeros((4, 4))

    for i in range(n):
        # Python: q(i,:)' * q(i,:) -> np.outer(q[i,:], q[i,:])
        M = M + w[i] * np.outer(q[i, :], q[i, :])

    # Python: [u, ~] = eig(M) -> eigenvalues, eigenvectors
    eigenvalues, u = np.linalg.eig(M)

    # Python: u(:,4) -> eigenvector corresponding to largest eigenvalue
    # Note: np.linalg.eig does not sort eigenvalues, so we need to find the max
    max_idx = np.argmax(eigenvalues)
    qAveraged = u[:, max_idx].real

    return qAveraged


# %[appendix]{"version":"1.0"}


def mean_angle(theta_array: np.ndarray, w: np.ndarray) -> float:
    """
    # Mean angle calculation with weights

    Parameters
    ----------
    theta_array : np.ndarray
        array of angles, rad
    w : np.ndarray
        weights

    Returns
    -------
    thetaAve : float
        mean angle, rad
    """
    theta_array = np.atleast_1d(theta_array).flatten()
    w = np.atleast_1d(w).flatten()
    w = w / np.sum(w)

    x = np.sum(w * np.cos(theta_array))
    y = np.sum(w * np.sin(theta_array))

    thetaAve = np.arctan2(y, x)

    return thetaAve


def q_err(scalar: int, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """
    # calculate quaternion error
    q_dを回転させてqに一致させるためのerror quaternion

    Parameters
    ----------
    scalar : int
        specifies the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    q : np.ndarray
        quaternions, nx4 matrix
    qd : np.ndarray
        desired (or estimated) quaternion, nx4 matrix

    Returns
    -------
    qe : np.ndarray
        quaternion error, nx4 matrix
        q_e = q_d^{-1} (circle) q = q (otimes) q_d^{-1}

    Notes
    -----
    NA

    References
    ----------
    Markley, F. L., and Crassidis, J. L., Fundamentals of Spacecraft Attitude
    Determination and Control, New York, NY: Springer, 2014.

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q_conj, q_inv
    """
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # q \otimes p = [ q0 * p0 - qv' * pv
    #               q0 .* pv + p0 .* qv - cross(qv,pv)]

    qInvD = q_inv(scalar, qd)

    qe = q_mult(scalar, 1, q, qInvD)

    return qe


# %[appendix]{"version":"1.0"}


def q_rotation(*args, scalar: int = 4) -> np.ndarray:
    """
    # rotating coordinate frames with quaternions
    ベクトルを回転後の座標系で表示（ベクトルを回転させているわけではない）

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    r : np.ndarray
        vector, nx3 vector
    q : np.ndarray
        quaternions, nx4 vector

    Returns
    -------
    rb : np.ndarray
        vector expressed with the rotated coordinate frame, nx3 matrix

    Notes
    -----
    r_b = q (otimes) r (otimes) q^{-1} = q^{-1} (circle) r (circle) q

    References
    ----------
    NA

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q_inv
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     r (:,3) {mustBeNumeric}
    #     q (:,4) {mustBeNumeric}
    # end
    if len(args) == 3:
        if isinstance(args[0], (int, np.integer)) and args[0] in [0, 4]:
            scalar, r, q = args
        else:
            r, q, third = args
            if isinstance(third, (int, np.integer)) and third in [0, 4]:
                scalar = int(third)
            else:
                raise TypeError("q_rotation third positional argument must be scalar(0/4)")
    elif len(args) == 2:
        r, q = args
    else:
        raise TypeError("q_rotation expects (r, q) or (scalar, r, q)")

    assert scalar in [0, 4], "quaternion definition is unclear"

    # Python: Ensure inputs are 2D for consistent indexing
    r = np.atleast_2d(r)
    q = np.atleast_2d(q)

    n = q.shape[0]
    rTmp = np.column_stack([np.zeros((n, 1)), r])

    # q0がスカラーの定義かつ \odot のクォータニオン積の定義で計算する
    if scalar == 0:
        pass  # do nothing
    elif scalar == 4:
        q = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
    else:
        raise ValueError('quaternion definition is unclear')

    qInverse = q_inv(0, q)

    tmp = q_mult(0, 0, qInverse, rTmp)
    tmp2 = q_mult(0, 0, tmp, q)

    rb = tmp2[:, 1:4]

    return rb


# %[appendix]{"version":"1.0"}


def triad(v1_i: np.ndarray, v2_i: np.ndarray,
          w1_b: np.ndarray, w2_b: np.ndarray) -> np.ndarray:
    """
    # calculate directional cosine matrix (DCM) using TRIAD method

    Parameters
    ----------
    v1_i : np.ndarray
        reference vectors expressed with inertial frame, nx3 matrix
    v2_i : np.ndarray
        reference vectors expressed with inertial frame, nx3 matrix
    w1_b : np.ndarray
        reference vectors expressed with body-fixed frame, nx3 matrix
    w2_b : np.ndarray
        reference vectors expressed with body-fixed frame, nx3 matrix

    Returns
    -------
    R : np.ndarray
        directional cosine matrix (rotation matrix) from inertial frame
        to body-fixed frame, 3 x 3 x n array

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20220201  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm, dcm2q
    """
    # arguments
    #     v1_i (:,3) {mustBeNumeric}
    #     v2_i (:,3) {mustBeNumeric}
    #     w1_b (:,3) {mustBeNumeric}
    #     w2_b (:,3) {mustBeNumeric}
    # end

    # Python: Ensure inputs are 2D for consistent indexing
    v1_i = np.atleast_2d(v1_i)
    v2_i = np.atleast_2d(v2_i)
    w1_b = np.atleast_2d(w1_b)
    w2_b = np.atleast_2d(w2_b)

    # normalization
    # Python: vecnorm(v1_i, 2, 2) -> np.linalg.norm(v1_i, axis=1, keepdims=True)
    v1_i = v1_i / np.linalg.norm(v1_i, axis=1, keepdims=True)
    v2_i = v2_i / np.linalg.norm(v2_i, axis=1, keepdims=True)
    w1_b = w1_b / np.linalg.norm(w1_b, axis=1, keepdims=True)
    w2_b = w2_b / np.linalg.norm(w2_b, axis=1, keepdims=True)

    n = v1_i.shape[0]

    # pre-allocation
    # Python: zeros(3,3,n) -> (3, 3, n) array
    R = np.zeros((3, 3, n))

    for i in range(n):
        r1 = v1_i[i, :]
        r2 = np.cross(v1_i[i, :], v2_i[i, :])
        r3 = np.cross(r1, r2)

        s1 = w1_b[i, :]
        s2 = np.cross(w1_b[i, :], w2_b[i, :])
        s3 = np.cross(s1, s2)

        # Python: [s1' s2' s3'] * [r1' r2' r3']' -> column vectors, then matrix mult
        S = np.column_stack([s1, s2, s3])
        Rmat = np.column_stack([r1, r2, r3])
        R[:, :, i] = S @ Rmat.T

    return R


# %[appendix]{"version":"1.0"}
