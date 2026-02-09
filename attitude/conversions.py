"""
Conversion functions between different attitude representations
Python conversion from yMATLAB/attitude/
"""

import numpy as np


def q1axis(scalar: int, axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    # quaternion around a single axis
    1軸周りのquaternionを計算

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    axis : np.ndarray
        rotation axis, unit vector, nx3 matrix
    theta : np.ndarray
        rotation angle, rad, scalar, nx1 vector

    Returns
    -------
    q : np.ndarray
        quaternion, nx4 matrix

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
    dcm1axis
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar,[0, 4])}
    #     axis (:,3) {mustBeNumeric}
    #     theta (:,1) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure inputs are 2D for consistent indexing
    axis = np.atleast_2d(axis)
    theta = np.atleast_1d(theta).reshape(-1, 1)

    # normalize
    # Python: vecnorm(axis,2,2) -> np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)

    # q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T で一旦計算
    q = np.column_stack([axis * np.sin(theta / 2), np.cos(theta / 2)])

    if scalar == 0:
        q = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])

    return q


# %[appendix]{"version":"1.0"}


def q_axis_angle(scalar: int, q: np.ndarray):
    """
    # Euler axis and rotation angle from quaternions

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

    Returns
    -------
    eAxis : np.ndarray
        Euler axis, e, nx3 matrix
    eAngle : np.ndarray
        rotation angle, theta in [0, 2*pi), nx1 matrix, rad

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
    q_conj, q_inv
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure q is 2D for consistent indexing
    q = np.atleast_2d(q)

    if scalar == 0:
        eAngle = 2.0 * np.arccos(q[:, 0])
        eAxis = q[:, 1:4] / np.sin(eAngle / 2.0).reshape(-1, 1)

    elif scalar == 4:
        eAngle = 2.0 * np.arccos(q[:, 3])
        eAxis = q[:, 0:3] / np.sin(eAngle / 2.0).reshape(-1, 1)

    else:
        raise ValueError('definition of quaternions is unclear')

    eAngle = np.mod(eAngle, 2*np.pi)

    return eAxis, eAngle


# %[appendix]{"version":"1.0"}


def q2rot_vec(scalar: int, q: np.ndarray) -> np.ndarray:
    """
    # calculating rotation vector from quaternions

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
    rotVec : np.ndarray
        rotation vector, theta * e, nx3 matrix

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
    q2zyx
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    eAxis, theta = q_axis_angle(scalar, q)

    rotVec = theta.reshape(-1, 1) * eAxis

    return rotVec


# %[appendix]{"version":"1.0"}


def rot_vec2q(scalar: int, rv: np.ndarray) -> np.ndarray:
    """
    # calculating quaternion from rotation vectors

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    rv : np.ndarray
        rotation vector, theta * e, nx3 matrix

    Returns
    -------
    q : np.ndarray
        quaternions, nx4 matrix

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
    q2zyx
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     rv (:,3) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "quaternion definition is unclear"

    # Python: Ensure rv is 2D for consistent indexing
    rv = np.atleast_2d(rv)

    # Python: vecnorm(rv, 2, 2) -> np.linalg.norm(rv, axis=1, keepdims=True)
    theta = np.linalg.norm(rv, axis=1, keepdims=True)
    eAxis = rv / theta

    if scalar == 0:
        q = np.column_stack([np.cos(theta / 2), eAxis * np.sin(theta / 2)])
    elif scalar == 4:
        q = np.column_stack([eAxis * np.sin(theta / 2), np.cos(theta / 2)])
    else:
        raise ValueError('quaternion definition is unclear')

    return q


# %[appendix]{"version":"1.0"}


def q2grp(scalar: int, f: float, a: float, q: np.ndarray) -> np.ndarray:
    """
    # converting quaternions to generalized Rodrigues parameters

    Parameters
    ----------
    scalar : int
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    f : float
        scaling parameter for Rodrigues parameters, scalar
    a : float
        parameter for Rodrigues parameters, scalar
    q : np.ndarray
        quaternions, nx4 matrix

    Returns
    -------
    p : np.ndarray
        generalized Rodrigues parameters, nx3 matrix

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
    grp2q
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     f (1,1) {mustBeNumeric}
    #     a (1,1) {mustBeNumeric}
    #     q (:,4) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure q is 2D for consistent indexing
    q = np.atleast_2d(q)

    # q4 = scalar partとして計算．
    if scalar == 0:
        q = np.column_stack([q[:, 1:4], q[:, 0]])

    p = f * np.sign(q[:, 3]).reshape(-1, 1) * q[:, 0:3] / (a + np.abs(q[:, 3]).reshape(-1, 1))

    return p


# %[appendix]{"version":"1.0"}


def grp2q(scalar: int, f: float, a: float, p: np.ndarray) -> np.ndarray:
    """
    # converting generalized Rodrigues parameters to quaternions

    Parameters
    ----------
    scalar : int
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    f : float
        scaling parameter for Rodrigues parameters, scalar
    a : float
        parameter for Rodrigues parameters, scalar
    p : np.ndarray
        generalized Rodrigues parameters, nx3 matrix

    Returns
    -------
    q : np.ndarray
        quaternions, nx4 matrix

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
    q2grp
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     f (1,1) {mustBeNumeric}
    #     a (1,1) {mustBeNumeric}
    #     p (:,3) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure p is 2D for consistent indexing
    p = np.atleast_2d(p)

    # Python: vecnorm(p, 2, 2) -> np.linalg.norm(p, axis=1, keepdims=True)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)

    q4Num = -a * p_norm**2 + f * np.sqrt(f**2 + (1 - a**2) * p_norm**2)
    q4Den = f**2 + p_norm**2

    q4 = q4Num / q4Den

    qv = (a + q4) * p / f

    if scalar == 0:
        q = np.column_stack([q4, qv])
    else:
        q = np.column_stack([qv, q4])

    return q


# %[appendix]{"version":"1.0"}


def q2rodrigues(scalar: int, q: np.ndarray) -> np.ndarray:
    """
    # calculating Rodrigues parameters from quaternions

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
    Rod : np.ndarray
        Rodrigues parameters, nx3 matrix

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

    f = 1
    a = 0

    Rod = q2grp(scalar, f, a, q)

    return Rod


# %[appendix]{"version":"1.0"}


def rodrigues2q(scalar: int, rod: np.ndarray) -> np.ndarray:
    """
    # calculating quaternions from Rodrigues parameters

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    rod : np.ndarray
        Rodrigues parameters, nx3 matrix

    Returns
    -------
    q : np.ndarray
        quaternions, nx4 matrix

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
    #     rod (:,3) {mustBeNumeric}
    # end

    f = 1
    a = 0

    q = grp2q(scalar, f, a, rod)

    return q


# %[appendix]{"version":"1.0"}
