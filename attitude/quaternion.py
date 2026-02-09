"""
Quaternion operations
Python conversion from yMATLAB/attitude/
"""

import numpy as np


def q_conj(*args, scalar: int = 4) -> np.ndarray:
    """
    # quaternion conjugate

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    q : np.ndarray
        quaternions, nx4 matrix

    Returns
    -------
    qC : np.ndarray
        quaternion conjugate, nx4 matrix
        q^dagger = [-q_v^T, q_4]^T

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
    q_inv
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4)
    # end
    if len(args) == 2:
        scalar, q = args
    elif len(args) == 1:
        q = args[0]
    else:
        raise TypeError("q_conj expects (q) or (scalar, q)")

    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure q is 2D for consistent indexing
    q = np.atleast_2d(q)

    if scalar == 0:
        qC = np.column_stack([q[:, 0], -q[:, 1:4]])

    elif scalar == 4:
        qC = np.column_stack([-q[:, 0:3], q[:, 3]])

    else:
        raise ValueError('definition of quaternions is unclear')

    return qC


# %[appendix]{"version":"1.0"}


def q_inv(*args, scalar: int = 4) -> np.ndarray:
    """
    # quaternion inverse

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    q : np.ndarray
        quaternions, nx4 matrix

    Returns
    -------
    qInv : np.ndarray
        quaternion inverse, nx4 matrix
        q^{-1} = q^* / ||q||^2  where q^* is quaternion conjugate

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
    q_conj
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    # end
    if len(args) == 2:
        scalar, q = args
    elif len(args) == 1:
        q = args[0]
    else:
        raise TypeError("q_inv expects (q) or (scalar, q)")

    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure q is 2D for consistent indexing
    q = np.atleast_2d(q)

    # Python: vecnorm(q,2,2) -> np.linalg.norm(q, axis=1, keepdims=True)
    norm_sq = np.linalg.norm(q, axis=1, keepdims=True) ** 2

    if scalar == 0:
        q_inverse = np.column_stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]]) / norm_sq

    elif scalar == 4:
        q_inverse = np.column_stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]]) / norm_sq

    else:
        raise ValueError('definition of quaternions is unclear')

    return q_inverse


# %[appendix]{"version":"1.0"}


def q_mult(*args, scalar: int = 4, definition: int = 1) -> np.ndarray:
    """
    # Quaternion multiplication

    Parameters
    ----------
    scalar : int
        specifies the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    definition : int
        specifies the definition of the quaternion multiplication
        def == 0:
            q (circle) p = [q4*p_v + p4*q_v + q_v x p_v; q4*p4 - q_v^T*p_v]
        def == 1:
            q (otimes) p = [q4*p_v + p4*q_v - q_v x p_v; q4*p4 - q_v^T*p_v]
    q : np.ndarray
        quaternions, nx4 matrix
    p : np.ndarray
        quaternions, nx4 matrix

    Returns
    -------
    output : np.ndarray
        quaternions, nx4 matrix

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
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     def (1,1) {mustBeMember(def, [0, 1])}
    #     q (:,4)
    #     p (:,4)
    # end
    if len(args) == 4:
        scalar, definition, q, p = args
    elif len(args) == 2:
        q, p = args
    elif len(args) == 3:
        q, p, third = args
        if isinstance(third, (int, np.integer)) and third in [0, 4]:
            scalar = int(third)
        elif isinstance(third, (int, np.integer)) and third in [0, 1]:
            definition = int(third)
        else:
            raise TypeError("q_mult third positional argument must be scalar(0/4) or definition(0/1)")
    else:
        raise TypeError("q_mult expects (q, p) or (scalar, definition, q, p)")

    assert scalar in [0, 4], "definition of quaternions is unclear"
    assert definition in [0, 1], "definition of quaternion multiplication is unclear"

    # Python: Ensure q and p are 2D for consistent indexing
    q = np.atleast_2d(q)
    p = np.atleast_2d(p)

    # 入力のサイズを取得
    nQ = q.shape[0]
    output = np.zeros((nQ, 4))

    if scalar == 0:

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        p0 = p[:, 0]
        p1 = p[:, 1]
        p2 = p[:, 2]
        p3 = p[:, 3]

        # スカラー部の計算
        scalarPart = q0 * p0 - (q1 * p1 + q2 * p2 + q3 * p3)

        # ベクトル部の計算（cross積を直接展開）
        # cross(qv, pv)
        cross1 = q2 * p3 - q3 * p2
        cross2 = q3 * p1 - q1 * p3
        cross3 = q1 * p2 - q2 * p1

        if definition == 0:

            output = np.column_stack([
                scalarPart,
                q0 * p1 + p0 * q1 + cross1,
                q0 * p2 + p0 * q2 + cross2,
                q0 * p3 + p0 * q3 + cross3
            ])

        else:  # def == 1
            output = np.column_stack([
                scalarPart,
                q0 * p1 + p0 * q1 - cross1,
                q0 * p2 + p0 * q2 - cross2,
                q0 * p3 + p0 * q3 - cross3
            ])

    else:  # scalar == 4
        q1 = q[:, 0]
        q2 = q[:, 1]
        q3 = q[:, 2]
        q4 = q[:, 3]

        p1 = p[:, 0]
        p2 = p[:, 1]
        p3 = p[:, 2]
        p4 = p[:, 3]

        scalarPart = q4 * p4 - (q1 * p1 + q2 * p2 + q3 * p3)

        # （cross積を直接展開 = cross(qv, pv) ）
        cross1 = p3 * q2 - p2 * q3
        cross2 = p1 * q3 - p3 * q1
        cross3 = p2 * q1 - p1 * q2

        if definition == 0:
            output = np.column_stack([
                q4 * p1 + p4 * q1 + cross1,
                q4 * p2 + p4 * q2 + cross2,
                q4 * p3 + p4 * q3 + cross3,
                scalarPart
            ])
        else:  # def == 1
            output = np.column_stack([
                q4 * p1 + p4 * q1 - cross1,
                q4 * p2 + p4 * q2 - cross2,
                q4 * p3 + p4 * q3 - cross3,
                scalarPart
            ])

    return output


# %[appendix]{"version":"1.0"}


def q2dcm(*args, scalar: int = 4) -> np.ndarray:
    """
    # calculating directional cosine matrix (DCM) from quaternions
    quaternionから回転行列を計算

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    q : np.ndarray
        quaternions, 1x4 vector

    Returns
    -------
    DCM : np.ndarray
        rotation matrix, 3x3 matrix

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
    q2zyx
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    # end
    if len(args) == 2:
        scalar, q = args
    elif len(args) == 1:
        q = args[0]
    else:
        raise TypeError("q2dcm expects (q) or (scalar, q)")

    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: q(1) -> q[0] (0-indexed)
    qNorm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    qTmp = q / qNorm  # normalize

    # calculate using q4 definition
    if scalar == 0:
        q4 = qTmp[0]
        q1 = qTmp[1]
        q2 = qTmp[2]
        q3 = qTmp[3]

    elif scalar == 4:
        q1 = qTmp[0]
        q2 = qTmp[1]
        q3 = qTmp[2]
        q4 = qTmp[3]

    DCM = np.array([
        [q1**2 - q2**2 - q3**2 + q4**2, 2.0 * (q1 * q2 + q4 * q3), 2.0 * (q1 * q3 - q4 * q2)],
        [2.0 * (q1 * q2 - q4 * q3), -q1**2 + q2**2 - q3**2 + q4**2, 2.0 * (q2 * q3 + q4 * q1)],
        [2.0 * (q1 * q3 + q4 * q2), 2.0 * (q2 * q3 - q4 * q1), -q1**2 - q2**2 + q3**2 + q4**2]
    ])

    return DCM


# %[appendix]{"version":"1.0"}


def dcm2q(*args, scalar: int = 4) -> np.ndarray:
    """
    # calculating quaternions from direction cosine matrix (DCM)
    回転行列からquaternionを計算
    ただしこの方法だとquaternionが不連続になるので注意．連続性も確保したい場合は dcm2q_continuous を使う．

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    R : np.ndarray
        rotation matrix, 3x3 matrix

    Returns
    -------
    q : np.ndarray
        quaternions, 1x4 vector

    Notes
    -----
    NA

    References
    ----------
    Markley, F. L., "Unit Quaternion from Rotation Matrix," Journal of
    Guidance Control, and Dynamics, vol. 31, Mar. 2008, pp. 440-442.

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     R (3,3) {mustBeNumeric}
    # end
    if len(args) == 2:
        scalar, R = args
    elif len(args) == 1:
        R = args[0]
    else:
        raise TypeError("dcm2q expects (R) or (scalar, R)")

    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: R(1,1) -> R[0,0] (0-indexed)
    trR = R[0, 0] + R[1, 1] + R[2, 2]  # trace
    tmp = np.array([R[0, 0], R[1, 1], R[2, 2], trR])

    # Python: max returns (value, index), we need index (1-indexed in MATLAB -> 0-indexed in Python)
    iq = np.argmax(tmp)

    # Python: switch-case -> if-elif (iq is 0-indexed)
    if iq == 0:  # case 1 in MATLAB
        x1 = np.array([
            1 + 2*R[0, 0] - trR,
            R[0, 1] + R[1, 0],
            R[0, 2] + R[2, 0],
            R[1, 2] - R[2, 1]
        ])
        q = x1 / np.linalg.norm(x1)

    elif iq == 1:  # case 2 in MATLAB
        x2 = np.array([
            R[1, 0] + R[0, 1],
            1 + 2 * R[1, 1] - trR,
            R[1, 2] + R[2, 1],
            R[2, 0] - R[0, 2]
        ])
        q = x2 / np.linalg.norm(x2)

    elif iq == 2:  # case 3 in MATLAB
        x3 = np.array([
            R[2, 0] + R[0, 2],
            R[2, 1] + R[1, 2],
            1 + 2 * R[2, 2] - trR,
            R[0, 1] - R[1, 0]
        ])
        q = x3 / np.linalg.norm(x3)

    else:  # case 4 in MATLAB (iq == 3)
        x4 = np.array([
            R[1, 2] - R[2, 1],
            R[2, 0] - R[0, 2],
            R[0, 1] - R[1, 0],
            1 + trR
        ])
        q = x4 / np.linalg.norm(x4)

    # Python: q = q' not needed (already 1D array), just ensure it's 1x4
    # q is already in [q1, q2, q3, q4] format (scalar-last)

    if scalar == 0:
        q = np.array([q[3], q[0], q[1], q[2]])

    return q


# %[appendix]{"version":"1.0"}


def dcm2q_continuous(*args, scalar: int = 4) -> np.ndarray:
    """
    # calculating optimal continuous quaternion from direction cosine matrix (DCM)

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    qK : np.ndarray
        previous quaternion, q_{k-1}, 1x4 matrix
    dcm : np.ndarray
        direction cosine matrix, 3x3 matrix

    Returns
    -------
    q : np.ndarray
        quaternion, 1x4 matrix

    Notes
    -----
    NA

    References
    ----------
    Wu, Jin. "Optimal Continuous Unit Quaternions from Rotation Matrices."
    Journal of Guidance, Control, and Dynamics, vol. 42, no. 4, 2019, pp. 919-22.

    Revisions
    ---------
    20221008  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    dcm2q
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     qK (1,4) {mustBeNumeric}
    #     dcm (3,3) {mustBeNumeric}
    # end
    if len(args) == 3:
        scalar, qK, dcm = args
    elif len(args) == 2:
        qK, dcm = args
    else:
        raise TypeError("dcm2q_continuous expects (qK, dcm) or (scalar, qK, dcm)")

    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure qK is 1D array
    qK = np.atleast_1d(qK).flatten()

    # temporarily the definition that q4 = cos(theta/2) is used
    if scalar == 0:
        qK = np.array([qK[1], qK[2], qK[3], qK[0]])

    # Eq. (7)
    B = 1/3 * dcm
    z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])
    K = np.block([
        [B + B.T - np.trace(B) * np.eye(3), z.reshape(-1, 1)],
        [z.reshape(1, -1), np.array([[np.trace(B)]])]
    ])
    eigenvalues, V = np.linalg.eig(K)

    # Python: V * diag([0 0 0 1]) * V^(-1) * qK'
    q = V @ np.diag([0, 0, 0, 1]) @ np.linalg.inv(V) @ qK

    q = q / np.linalg.norm(q)

    if scalar == 0:
        q = np.array([q[3], q[0], q[1], q[2]])

    return q


# %[appendix]{"version":"1.0"}
