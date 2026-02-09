"""
DCM (Direction Cosine Matrix) related functions
Python conversion from yMATLAB/attitude/
"""

import numpy as np


def dcm1axis(axis: int, phi: float) -> np.ndarray:
    """
    # direction cosine matrix (DCM) about a single axis
    1軸周りの回転行列を計算

    Parameters
    ----------
    axis : int
        rotation axis, 1 == x-axis, 2 == y-axis, 3 == z-axis
    phi : float
        rotation angle, rad, scalar

    Returns
    -------
    R : np.ndarray
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
    q2dcm, dcm2q
    """
    # arguments
    #     axis (1,1) {mustBeMember(axis, [1, 2, 3])}
    #     phi (1,1)
    # end
    assert axis in [1, 2, 3], "axis must be 1, 2, or 3"

    cp = np.cos(phi)
    sp = np.sin(phi)

    if axis == 1:
        R = np.array([[1, 0, 0],
                      [0, cp, sp],
                      [0, -sp, cp]])
    elif axis == 2:
        R = np.array([[cp, 0, -sp],
                      [0, 1, 0],
                      [sp, 0, cp]])
    else:
        R = np.array([[cp, sp, 0],
                      [-sp, cp, 0],
                      [0, 0, 1]])

    return R


# %[appendix]{"version":"1.0"}


def dcm1axis_x(phi: float) -> np.ndarray:
    """
    # direction cosine matrix (DCM) about X axis
    x軸周りの回転行列を計算

    Parameters
    ----------
    phi : float
        rotation angle, rad, scalar

    Returns
    -------
    R : np.ndarray
        rotation matrix, 3x3 matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20241216  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm, dcm1axis
    """
    # arguments
    #     phi (1,1)
    # end

    cp = np.cos(phi)
    sp = np.sin(phi)

    R = np.array([[1, 0, 0],
                  [0, cp, sp],
                  [0, -sp, cp]])

    return R


# %[appendix]{"version":"1.0"}


def dcm1axis_y(phi: float) -> np.ndarray:
    """
    # direction cosine matrix (DCM) about Y axis
    y軸周りの回転行列を計算

    Parameters
    ----------
    phi : float
        rotation angle, rad, scalar

    Returns
    -------
    R : np.ndarray
        rotation matrix, 3x3 matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20241216  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm, dcm1axis
    """
    # arguments
    #     phi (1,1)
    # end

    cp = np.cos(phi)
    sp = np.sin(phi)

    R = np.array([[cp, 0, -sp],
                  [0, 1, 0],
                  [sp, 0, cp]])

    return R


# %[appendix]{"version":"1.0"}


def dcm1axis_z(phi: float) -> np.ndarray:
    """
    # direction cosine matrix (DCM) about Z axis
    z軸周りの回転行列を計算

    Parameters
    ----------
    phi : float
        rotation angle, rad, scalar

    Returns
    -------
    R : np.ndarray
        rotation matrix, 3x3 matrix

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20241216  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    q2dcm, dcm1axis
    """
    # arguments
    #     phi (1,1)
    # end

    cp = np.cos(phi)
    sp = np.sin(phi)

    R = np.array([[cp, sp, 0],
                  [-sp, cp, 0],
                  [0, 0, 1]])

    return R


# %[appendix]{"version":"1.0"}


def zyx2dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    # ZYX Euler angle to directional cosine matrix (rotation matrix)

    Parameters
    ----------
    phi : float
        rotation angle around Z-axis (1st rotation)
    theta : float
        rotation angle around Y-axis (2nd rotation)
    psi : float
        rotation angle around X-axis (3rd rotation)

    Returns
    -------
    R : np.ndarray
        directional cosine matrix (rotation matrix), 3x3

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
    q2dcm
    """
    # arguments
    #     phi (1,1)
    #     theta (1,1)
    #     psi (1,1)
    # end

    R = np.array([
        [np.cos(theta)*np.cos(phi),
         np.cos(theta)*np.sin(phi),
         -np.sin(theta)],
        [np.sin(theta)*np.cos(phi)*np.sin(psi) - np.sin(phi)*np.cos(psi),
         np.sin(theta)*np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi),
         np.cos(theta)*np.sin(psi)],
        [np.sin(theta)*np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi),
         np.sin(theta)*np.sin(phi)*np.cos(psi) - np.cos(phi)*np.sin(psi),
         np.cos(theta)*np.cos(psi)]
    ])

    return R


# %[appendix]{"version":"1.0"}


def zxz2dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    # ZXZ Euler angle to directional cosine matrix (rotation matrix)

    Parameters
    ----------
    phi : float
        rotation angle around Z-axis (1st rotation)
    theta : float
        rotation angle around X-axis (2nd rotation)
    psi : float
        rotation angle around Z-axis (3rd rotation)

    Returns
    -------
    R : np.ndarray
        directional cosine matrix (rotation matrix), 3x3

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
    q2dcm, zyx2dcm
    """
    # arguments
    #     phi (1,1) {mustBeNumeric}
    #     theta (1,1) {mustBeNumeric}
    #     psi (1,1) {mustBeNumeric}
    # end

    R = dcm1axis(3, psi) @ dcm1axis(1, theta) @ dcm1axis(3, phi)

    return R


# %[appendix]{"version":"1.0"}


def zyz2dcm(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    # ZYZ (3-2-3) Euler angle to directional cosine matrix (rotation matrix)

    Parameters
    ----------
    phi : float
        rotation angle around Z-axis (1st rotation)
    theta : float
        rotation angle around Y-axis (2nd rotation)
    psi : float
        rotation angle around Z-axis (3rd rotation)

    Returns
    -------
    R : np.ndarray
        directional cosine matrix (rotation matrix), 3x3

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
    q2dcm
    """
    # arguments
    #     phi (1,1)
    #     theta (1,1)
    #     psi (1,1)
    # end

    R = dcm1axis(3, psi) @ dcm1axis(2, theta) @ dcm1axis(3, phi)

    return R


# %[appendix]{"version":"1.0"}
