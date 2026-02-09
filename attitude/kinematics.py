"""
Kinematics functions for attitude dynamics
Python conversion from yMATLAB/attitude/
"""

import numpy as np
from scipy import integrate


def _skew(v: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix from a 3-element vector.

    Parameters
    ----------
    v : np.ndarray
        3-element vector

    Returns
    -------
    S : np.ndarray
        3x3 skew-symmetric matrix
    """
    v = np.atleast_1d(v).flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def q_kine(scalar: int, q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    # quaternion kinematics
    kinematic equation using quaternions:
    q_dot = 1/2 * omega (otimes) q

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
    w : np.ndarray
        angular velocity, 1x3 vector

    Returns
    -------
    qKine : np.ndarray
        quaternion derivative, 1x4 vector

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20200901  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    #     w (:,3) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    q = np.atleast_1d(q).flatten()
    w = np.atleast_1d(w).flatten()
    wx = w[0]
    wy = w[1]
    wz = w[2]

    if scalar == 0:
        mat = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        qKine = 0.5 * mat @ q

    elif scalar == 4:
        mat = np.array([
            [0, wz, -wy, wx],
            [-wz, 0, wx, wy],
            [wy, -wx, 0, wz],
            [-wx, -wy, -wz, 0]
        ])
        qKine = 0.5 * mat @ q

    else:
        raise ValueError('definition of quaternions is unclear')

    return qKine


# %[appendix]{"version":"1.0"}


def q_prop_mat(scalar: int, dt: float, w: np.ndarray) -> np.ndarray:
    """
    # discrete kinematics of quaternions

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    dt : float
        time span, scalar, s
    w : np.ndarray
        angular rate, 1x3 vector

    Returns
    -------
    qProp : np.ndarray
        quaternion propagation matrix, 4x4 matrix

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
    #     dt (1,1) {mustBeNumeric}
    #     w (1,3) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    w = np.atleast_1d(w).flatten()

    norm_w = np.linalg.norm(w)
    psi = np.sin(0.5 * norm_w * dt) * w / norm_w

    if scalar == 0:
        # Python: [cos(...), -psi'; psi, cos(...)*eye(3) - skew(psi)]
        cos_term = np.cos(0.5 * norm_w * dt)
        qProp = np.block([
            [cos_term, -psi.reshape(1, -1)],
            [psi.reshape(-1, 1), cos_term * np.eye(3) - _skew(psi)]
        ])
    elif scalar == 4:
        cos_term = np.cos(0.5 * norm_w * dt)
        qProp = np.block([
            [cos_term * np.eye(3) - _skew(psi), psi.reshape(-1, 1)],
            [-psi.reshape(1, -1), cos_term]
        ])
    else:
        raise ValueError('definition of quaternions is unclear')

    return qProp


# %[appendix]{"version":"1.0"}


def q_mult_mat(*args, scalar: int = 4, definition: int = 1) -> np.ndarray:
    """
    # Quaternion multiplication matrix

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
        quaternions, 1x4 matrix

    Returns
    -------
    qMat : np.ndarray
        4x4 matrix

    Notes
    -----
    NA

    References
    ----------
    Markley, F. L., and Crassidis, J. L., Fundamentals of Spacecraft Attitude
    Determination and Control, New York, NY: Springer, 2014.

    Revisions
    ---------
    20231219  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q_conj, q_inv
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     def (1,1) {mustBeMember(def, [0, 1])}
    #     q (:,4) {mustBeNumeric}
    # end
    if len(args) == 3:
        scalar, definition, q = args
    elif len(args) == 1:
        q = args[0]
    elif len(args) == 2:
        q, third = args
        if isinstance(third, (int, np.integer)) and third in [0, 4]:
            scalar = int(third)
        elif isinstance(third, (int, np.integer)) and third in [0, 1]:
            definition = int(third)
        else:
            raise TypeError("q_mult_mat second positional argument must be scalar(0/4) or definition(0/1)")
    else:
        raise TypeError("q_mult_mat expects (q) or (scalar, definition, q)")

    assert scalar in [0, 4], "definition of quaternions is unclear"
    assert definition in [0, 1], "definition of quaternion multiplication is unclear"

    q = np.atleast_1d(q).flatten()

    if scalar == 0:
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        qv = np.array([q1, q2, q3])

        if definition == 0:
            qMat = np.block([
                [q0, -qv.reshape(1, -1)],
                [qv.reshape(-1, 1), q0 * np.eye(3) + _skew(qv)]
            ])
        elif definition == 1:
            qMat = np.block([
                [q0, -qv.reshape(1, -1)],
                [qv.reshape(-1, 1), q0 * np.eye(3) - _skew(qv)]
            ])
        else:
            raise ValueError('definition of quaternion multiplication is unclear')

    elif scalar == 4:
        q4 = q[3]
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        qv = np.array([q1, q2, q3])

        if definition == 0:
            qMat = np.block([
                [q4 * np.eye(3) + _skew(qv), qv.reshape(-1, 1)],
                [-qv.reshape(1, -1), q4]
            ])
        elif definition == 1:
            qMat = np.block([
                [q4 * np.eye(3) - _skew(qv), qv.reshape(-1, 1)],
                [-qv.reshape(1, -1), q4]
            ])
        else:
            raise ValueError('definition of quaternion multiplication is unclear')

    else:
        raise ValueError('definition of quaternions is unclear')

    return qMat


# %[appendix]{"version":"1.0"}


def rot_period(MOI: np.ndarray, w: np.ndarray) -> float:
    """
    # Rotation period of free rotational motion of a rigid body

    Parameters
    ----------
    MOI : np.ndarray
        principal axes of moment of inertia, kgm^2, 3x3 matrix
    w : np.ndarray
        angular rate, rad/s, 1x3 vector

    Returns
    -------
    T : float
        period of free rotation, s

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20190121  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    NA
    """
    # arguments
    #     MOI (3,3)
    #     w (1,3)
    # end

    w = np.atleast_1d(w).flatten()  # make column vector

    Jx = MOI[0, 0]
    Jy = MOI[1, 1]
    Jz = MOI[2, 2]

    E = 0.5 * w @ MOI @ w
    h = MOI @ w
    h2 = h @ h

    m = (Jy - Jx) * (2*Jz * E - h2) / (Jz - Jy) / (h2 - 2*Jx*E)

    def myfun(xi):
        return 1.0 / np.sqrt(1 - m * (np.sin(xi))**2)

    K, _ = integrate.quad(myfun, 0, np.pi/2)

    kappa = np.sqrt(Jx * (Jz - Jx) / Jy / (Jz - Jy))
    w3m = np.sqrt((h2 - 2*Jx*E) / Jz / (Jz - Jx))

    wp = ((Jy - Jz) / Jx) * kappa * w3m
    T = np.abs(4 * K / wp)

    return T


# %[appendix]{"version":"1.0"}
