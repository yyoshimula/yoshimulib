"""
Euler angle related functions
Python conversion from yMATLAB/attitude/
"""

import numpy as np
from .quaternion import q_mult
from .dcm import dcm1axis


def zyx2q(*args, scalar: int = 4) -> np.ndarray:
    """
    # ZYX Euler angle to quaternion

    Parameters
    ----------
    scalar : int
        specifies the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    phi : np.ndarray
        rotation angle around Z-axis (1st rotation), nx1
    theta : np.ndarray
        rotation angle around Y-axis (2nd rotation), nx1
    psi : np.ndarray
        rotation angle around X-axis (3rd rotation), nx1

    Returns
    -------
    q : np.ndarray
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
    q_mult
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     phi (:,1) {mustBeNumeric}
    #     theta (:,1) {mustBeNumeric}
    #     psi (:,1) {mustBeNumeric}
    # end
    if len(args) == 4:
        if isinstance(args[0], (int, np.integer)) and args[0] in [0, 4]:
            scalar, phi, theta, psi = args
        else:
            phi, theta, psi, third = args
            if isinstance(third, (int, np.integer)) and third in [0, 4]:
                scalar = int(third)
            else:
                raise TypeError("zyx2q fourth positional argument must be scalar(0/4)")
    elif len(args) == 3:
        phi, theta, psi = args
    else:
        raise TypeError("zyx2q expects (phi, theta, psi) or (scalar, phi, theta, psi)")

    assert scalar in [0, 4], "quaternion definition is unclear"

    # Python: Ensure inputs are 1D arrays
    phi = np.atleast_1d(phi).flatten()
    theta = np.atleast_1d(theta).flatten()
    psi = np.atleast_1d(psi).flatten()

    n = len(phi)

    # ### each quaternion around Z-axis, Y-axis, and X-axis
    # q4: scalar partで計算
    qPhi = np.column_stack([np.zeros((n, 2)), np.sin(phi/2), np.cos(phi/2)])
    qTheta = np.column_stack([np.zeros((n, 1)), np.sin(theta/2), np.zeros((n, 1)), np.cos(theta/2)])
    qPsi = np.column_stack([np.sin(psi/2), np.zeros((n, 2)), np.cos(psi/2)])
    q = q_mult(4, 1, qPsi, q_mult(4, 1, qTheta, qPhi))

    # ## quaternion
    if scalar == 0:
        q = np.column_stack([q[:, 3], q[:, 0:3]])

    elif scalar == 4:
        pass  # do nothing

    else:
        raise ValueError('quaternion definition is unclear')

    return q


# %[appendix]{"version":"1.0"}


def zxz2q(*args, scalar: int = 4) -> np.ndarray:
    """
    # ZXZ Euler angle to quaternion

    Parameters
    ----------
    scalar : int
        specifies the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    phi : np.ndarray
        rotation angle around Z-axis (1st rotation), nx1
    theta : np.ndarray
        rotation angle around X-axis (2nd rotation), nx1
    psi : np.ndarray
        rotation angle around Z-axis (3rd rotation), nx1

    Returns
    -------
    q : np.ndarray
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
    q_mult
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     phi (:,1) {mustBeNumeric}
    #     theta (:,1) {mustBeNumeric}
    #     psi (:,1) {mustBeNumeric}
    # end
    if len(args) == 4:
        if isinstance(args[0], (int, np.integer)) and args[0] in [0, 4]:
            scalar, phi, theta, psi = args
        else:
            phi, theta, psi, third = args
            if isinstance(third, (int, np.integer)) and third in [0, 4]:
                scalar = int(third)
            else:
                raise TypeError("zxz2q fourth positional argument must be scalar(0/4)")
    elif len(args) == 3:
        phi, theta, psi = args
    else:
        raise TypeError("zxz2q expects (phi, theta, psi) or (scalar, phi, theta, psi)")

    assert scalar in [0, 4], "quaternion definition is unclear"

    # Python: Ensure inputs are 1D arrays
    phi = np.atleast_1d(phi).flatten()
    theta = np.atleast_1d(theta).flatten()
    psi = np.atleast_1d(psi).flatten()

    n = len(phi)

    # ### each quaternion around Z-axis, X-axis, and Z-axis
    q1 = np.column_stack([np.zeros((n, 2)), np.sin(phi/2), np.cos(phi/2)])  # nx4 matrix
    q2 = np.column_stack([np.sin(theta/2), np.zeros((n, 2)), np.cos(theta/2)])
    q3 = np.column_stack([np.zeros((n, 2)), np.sin(psi/2), np.cos(psi/2)])
    tmp = q_mult(4, 1, q3, q_mult(4, 1, q2, q1))  # nx4 matrix

    # ## quaternion
    if scalar == 0:
        q = np.column_stack([tmp[:, 3], tmp[:, 0:3]])

    elif scalar == 4:
        q = tmp

    else:
        raise ValueError('quaternion definition is unclear')

    return q


# %[appendix]{"version":"1.0"}


def zyz2q(*args, scalar: int = 4) -> np.ndarray:
    """
    # ZYZ (3-2-3) Euler angle to quaternion

    Parameters
    ----------
    scalar : int
        specifies the definition of the quaternion
        scalar == 0:
            q = [q_0, q_1, q_2, q_3]^T = [cos(theta/2), e^T sin(theta/2)]^T
        scalar == 4:
            q = [q_1, q_2, q_3, q_4]^T = [e^T sin(theta/2), cos(theta/2)]^T
    phi : np.ndarray
        rotation angle around Z-axis (1st rotation), nx1
    theta : np.ndarray
        rotation angle around Y-axis (2nd rotation), nx1
    psi : np.ndarray
        rotation angle around Z-axis (3rd rotation), nx1

    Returns
    -------
    q : np.ndarray
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
    20240315  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q_mult, zyx2q
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     phi (:,1) {mustBeNumeric}
    #     theta (:,1) {mustBeNumeric}
    #     psi (:,1) {mustBeNumeric}
    # end
    if len(args) == 4:
        if isinstance(args[0], (int, np.integer)) and args[0] in [0, 4]:
            scalar, phi, theta, psi = args
        else:
            phi, theta, psi, third = args
            if isinstance(third, (int, np.integer)) and third in [0, 4]:
                scalar = int(third)
            else:
                raise TypeError("zyz2q fourth positional argument must be scalar(0/4)")
    elif len(args) == 3:
        phi, theta, psi = args
    else:
        raise TypeError("zyz2q expects (phi, theta, psi) or (scalar, phi, theta, psi)")

    assert scalar in [0, 4], "quaternion definition is unclear"

    # Python: Ensure inputs are 1D arrays
    phi = np.atleast_1d(phi).flatten()
    theta = np.atleast_1d(theta).flatten()
    psi = np.atleast_1d(psi).flatten()

    n = len(phi)

    # ### each quaternion around Z-axis, Y-axis, and Z-axis
    # q4: scalar partで計算
    qPhi = np.column_stack([np.zeros((n, 2)), np.sin(phi/2), np.cos(phi/2)])
    qTheta = np.column_stack([np.zeros((n, 1)), np.sin(theta/2), np.zeros((n, 1)), np.cos(theta/2)])
    qPsi = np.column_stack([np.zeros((n, 2)), np.sin(psi/2), np.cos(psi/2)])
    q = q_mult(4, 1, qPsi, q_mult(4, 1, qTheta, qPhi))

    # ## quaternion
    if scalar == 0:
        q = np.column_stack([q[:, 3], q[:, 0:3]])

    elif scalar == 4:
        pass  # do nothing

    else:
        raise ValueError('quaternion definition is unclear')

    return q


# %[appendix]{"version":"1.0"}


def q2zyx(scalar: int, q: np.ndarray) -> np.ndarray:
    """
    # calculating ZYX Euler angles from quaternions

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
    euler : np.ndarray
        Euler angles, [phi, theta, psi], rad, nx3 matrix
        where phi: 1st rotation around z-axis
              theta: 2nd rotation around y-axis
              psi: 3rd rotation around x-axis

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
    q2zyz
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure q is 2D for consistent indexing
    q = np.atleast_2d(q)

    # calculate DCM using the definition that q0 = cos(theta/2)
    if scalar == 4:
        q = np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])

    R23 = 2.0 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    R33 = q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2
    R12 = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R11 = q[:, 0]**2 + q[:, 1]**2 - q[:, 2]**2 - q[:, 3]**2
    R13 = 2.0 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])

    psi = np.arctan2(R23, R33)
    phi = np.arctan2(R12, R11)
    theta = np.arctan2(-R13, np.sqrt(R23**2 + R33**2))

    euler = np.column_stack([phi, theta, psi])  # [rad]

    return euler


# %[appendix]{"version":"1.0"}


def q2zyz(scalar: int, q: np.ndarray) -> np.ndarray:
    """
    # calculating ZYZ (3-2-3) Euler angles from quaternions

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
    euler : np.ndarray
        Euler angles, [phi, theta, psi], rad, nx3 matrix
        where phi: 1st rotation around z-axis
              theta: 2nd rotation around y-axis
              psi: 3rd rotation around z-axis

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20230809  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2zyx
    """
    # arguments
    #     scalar (1,1) {mustBeMember(scalar, [0, 4])}
    #     q (:,4) {mustBeNumeric}
    # end
    assert scalar in [0, 4], "definition of quaternions is unclear"

    # Python: Ensure q is 2D for consistent indexing
    q = np.atleast_2d(q)

    # calculate DCM using the definition that q4 = cos(theta/2)
    if scalar == 0:
        q = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])

    R23 = 2.0 * (q[:, 1] * q[:, 2] + q[:, 3] * q[:, 0])
    R13 = 2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])

    R31 = 2.0 * (q[:, 0] * q[:, 2] + q[:, 3] * q[:, 1])
    R32 = 2.0 * (q[:, 1] * q[:, 2] - q[:, 3] * q[:, 0])

    R33 = -q[:, 0]**2 - q[:, 1]**2 + q[:, 2]**2 + q[:, 3]**2

    psi = np.arctan2(R23, -R13)
    phi = np.arctan2(R32, R31)
    theta = np.arctan2(np.sqrt(R31**2 + R32**2), R33)

    euler = np.column_stack([phi, theta, psi])  # [rad]

    return euler


# %[appendix]{"version":"1.0"}


def euler_dcm(axis1: int, axis2: int, axis3: int,
              phi: float, theta: float, psi: float) -> np.ndarray:
    """
    # Euler angle to directional cosine matrix (rotation matrix)

    Parameters
    ----------
    axis1 : int
        1st rotation axis, must be 1, 2, or 3
    axis2 : int
        2nd rotation axis, must be 1, 2, or 3
    axis3 : int
        3rd rotation axis, must be 1, 2, or 3
    phi : float
        rotation angle around the 1st axis (1st rotation)
    theta : float
        rotation angle around 2nd axis (2nd rotation)
    psi : float
        rotation angle around 3rd axis (3rd rotation)

    Returns
    -------
    R : np.ndarray
        directional cosine matrix (rotation matrix), 3x3

    Notes
    -----
    if rotation matrix of ZYX (3-2-1) Euler angle is required:
        phi = np.deg2rad(30)
        theta = np.deg2rad(20)
        psi = np.deg2rad(10)
        R = euler_dcm(3, 2, 1, phi, theta, psi)

    References
    ----------
    Markley, F. L., and Crassidis, J. L., Fundamentals of Spacecraft Attitude
    Determination and Control, New York, NY: Springer, 2014.

    Revisions
    ---------
    20150101  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm
    """
    # arguments
    #     axis1 (1,1) {mustBeMember(axis1, [1, 2, 3])}
    #     axis2 (1,1) {mustBeMember(axis2, [1, 2, 3])}
    #     axis3 (1,1) {mustBeMember(axis3, [1, 2, 3])}
    #     phi (1,1)
    #     theta (1,1)
    #     psi (1,1)
    # end
    assert axis1 in [1, 2, 3], "axis1 must be 1, 2, or 3"
    assert axis2 in [1, 2, 3], "axis2 must be 1, 2, or 3"
    assert axis3 in [1, 2, 3], "axis3 must be 1, 2, or 3"

    R = dcm1axis(axis3, psi) @ dcm1axis(axis2, theta) @ dcm1axis(axis1, phi)

    return R


# %[appendix]{"version":"1.0"}


def generate_euler_angle_kinematics(first: int, second: int, third: int):
    """
    # Euler angle kinematics generator

    This function provides the kinematics matrix B for Euler angle rates.

    Parameters
    ----------
    first : int
        1st rotation axis, must be 1, 2, or 3
    second : int
        2nd rotation axis, must be 1, 2, or 3
    third : int
        3rd rotation axis, must be 1, 2, or 3

    Returns
    -------
    B_func : callable
        A function B(theta, psi) that returns the 3x3 kinematics matrix
        such that [phi_dot, theta_dot, psi_dot]^T = B(theta, psi) @ [wx, wy, wz]^T

    Notes
    -----
    This is a simplified version. The original MATLAB code uses symbolic
    computation to derive the matrix B analytically. This Python version
    provides the commonly used ZYX (3-2-1) case.

    For general Euler sequences, the derivation requires symbolic math.

    References
    ----------
    Markley, F. L., and Crassidis, J. L., Fundamentals of Spacecraft Attitude
    Determination and Control, New York, NY: Springer, 2014.

    Revisions
    ---------
    20231212  y.yoshimura, y.yoshimula@gmail.com
    Python conversion note: Only ZYX (3-2-1) sequence is implemented numerically.
    """
    # Python: This is a simplified implementation for the ZYX case
    # The original MATLAB script uses symbolic computation

    if first == 1 and second == 2 and third == 3:
        # ZYX (3-2-1) Euler angle kinematics
        # [phi_dot, theta_dot, psi_dot]^T = B(theta, psi) @ omega
        def B_func(theta, psi):
            ct = np.cos(theta)
            st = np.sin(theta)
            cp = np.cos(psi)
            sp = np.sin(psi)

            B = np.array([
                [cp / ct, -sp / ct, 0],
                [sp, cp, 0],
                [-cp * st / ct, sp * st / ct, 1]
            ])
            return B

        return B_func

    else:
        raise NotImplementedError(
            f"Euler sequence ({first}, {second}, {third}) is not implemented. "
            "Only ZYX (1, 2, 3) is available. "
            "For other sequences, symbolic derivation is required."
        )


# %[appendix]{"version":"1.0"}
