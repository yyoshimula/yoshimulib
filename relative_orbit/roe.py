"""
Relative Orbital Elements (ROE) conversions
Python conversion from yMATLAB/relativeOrbit/
"""

import numpy as np
from ..orbit.kepler import mean_anomaly
from ..math_utils import wrap_pi


def oe2roe(chief: np.ndarray, deputy: np.ndarray, anomaly_flag: int) -> np.ndarray:
    """
    # calculating relative orbital elements (ROE) from absolute orbital elements

    Parameters
    ----------
    chief : np.ndarray
        absolute orbital elements of chief, n x 6 matrix
        [a, e, i, RAAN, w, f (or M)] (km, -, rad, rad, rad, rad)
    deputy : np.ndarray
        absolute orbital elements of deputy, n x 6 matrix
        [a, e, i, RAAN, w, f (or M)] (km, -, rad, rad, rad, rad)
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly

    Returns
    -------
    roe : np.ndarray
        relative orbital elements (ROE), n x 6 matrix
        [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]

    Notes
    -----
    ROE definitions:
    delta_a = (a_d - a) / a
    delta_lambda = (u_d - u) + (RAAN_d - RAAN) * cos(i)
    delta_ex = e_xd - e_x
    delta_ey = e_yd - e_y
    delta_ix = i_d - i
    delta_iy = (RAAN_d - RAAN) * sin(i)

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura
    20261005  y.yoshimura, fix: RAAN difference wrapped to [-pi, pi) (chief and
              deputy RAAN on both sides of 0/2pi)

    See also
    --------
    roe2deputy_oe
    """
    assert anomaly_flag in [0, 1], "anomaly_flag must be 0 or 1"

    # copies: the inputs must not be modified
    chief = np.array(chief, dtype=float, ndmin=2)
    deputy = np.array(deputy, dtype=float, ndmin=2)

    # Wrap RAAN, w, f/M to [0, 2pi)
    chief[:, 3:6] = np.mod(chief[:, 3:6], 2 * np.pi)
    deputy[:, 3:6] = np.mod(deputy[:, 3:6], 2 * np.pi)

    if anomaly_flag == 1:
        m_c = mean_anomaly(chief[:, 1], chief[:, 5])
        m_d = mean_anomaly(deputy[:, 1], deputy[:, 5])
    else:
        m_c = chief[:, 5]
        m_d = deputy[:, 5]

    # Chief parameters
    a_c = chief[:, 0]
    u_c = np.mod(chief[:, 4] + m_c, 2 * np.pi)  # mean argument of latitude
    ex_c = chief[:, 1] * np.cos(chief[:, 4])
    ey_c = chief[:, 1] * np.sin(chief[:, 4])
    inc_c = chief[:, 2]
    raan_c = chief[:, 3]

    # Deputy parameters
    a_d = deputy[:, 0]
    u_d = np.mod(deputy[:, 4] + m_d, 2 * np.pi)
    ex_d = deputy[:, 1] * np.cos(deputy[:, 4])
    ey_d = deputy[:, 1] * np.sin(deputy[:, 4])
    inc_d = deputy[:, 2]
    raan_d = deputy[:, 3]

    # ROEs
    # RAAN difference wrapped to [-pi, pi): raan_c and raan_d are in [0, 2pi), so
    # their plain difference jumps by 2pi when the two straddle 0/2pi
    d_raan = wrap_pi(raan_d - raan_c)

    delta_a = (a_d - a_c) / a_c
    delta_lambda = u_d - u_c + d_raan * np.cos(inc_c)
    delta_ex = ex_d - ex_c
    delta_ey = ey_d - ey_c
    delta_ix = inc_d - inc_c
    delta_iy = d_raan * np.sin(inc_c)

    delta_lambda = wrap_pi(delta_lambda)

    roe = np.column_stack([delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy])

    return roe


def roe2rtn(roe: np.ndarray, chief_oe: np.ndarray, anomaly_flag: int,
            mu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    # Mapping relative orbital elements (ROEs) to relative position and velocity at RTN frame

    Parameters
    ----------
    roe : np.ndarray
        relative orbital elements (ROE), n x 6 matrix
        [delta_a, delta_lambda, delta_ex, delta_ey, delta_ix, delta_iy]
    chief_oe : np.ndarray
        absolute orbital elements of chief, n x 6 matrix
        [a, e, i, RAAN, w, f (or M)] (m or km, -, rad, rad, rad, rad)
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly
    mu : float
        gravitational constant (unit must be unified with position and velocity)

    Returns
    -------
    x_rtn : np.ndarray
        deputy position @ RTN frame, m or km, n x 3
    v_rtn : np.ndarray
        deputy velocity @ RTN frame, m or km/s, n x 3

    Notes
    -----
    重力定数と位置・速度は単位を合わせること

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com
    """
    assert anomaly_flag in [0, 1], "anomaly_flag must be 0 or 1"

    roe = np.atleast_2d(roe)
    chief_oe = np.atleast_2d(chief_oe)

    roe[:, 1] = np.arctan2(np.sin(roe[:, 1]), np.cos(roe[:, 1]))
    chief_oe[:, 5] = np.mod(chief_oe[:, 5], 2 * np.pi)

    a_c = chief_oe[:, 0]  # semi-major axis of chief
    n_c = np.sqrt(mu / a_c ** 3)  # mean motion of chief

    if anomaly_flag == 1:
        m_c = mean_anomaly(chief_oe[:, 1], chief_oe[:, 5])
    else:
        m_c = chief_oe[:, 5]

    m_c = np.mod(m_c, 2 * np.pi)
    u_c = chief_oe[:, 4] + m_c  # mean argument of latitude, rad
    u_c = np.mod(u_c, 2 * np.pi)

    # Mapping
    x = roe[:, 0] - np.cos(u_c) * roe[:, 2] - np.sin(u_c) * roe[:, 3]
    y = roe[:, 1] + 2 * np.sin(u_c) * roe[:, 2] - 2 * np.cos(u_c) * roe[:, 3]

    vx = n_c * np.sin(u_c) * roe[:, 2] - n_c * np.cos(u_c) * roe[:, 3]
    vy = -3 / 2 * n_c * roe[:, 0] + 2 * n_c * np.cos(u_c) * roe[:, 2] + 2 * n_c * np.sin(u_c) * roe[:, 3]

    z = np.sin(u_c) * roe[:, 4] - np.cos(u_c) * roe[:, 5]
    vz = n_c * np.cos(u_c) * roe[:, 4] + n_c * np.sin(u_c) * roe[:, 5]

    x_rtn = a_c[:, np.newaxis] * np.column_stack([x, y, z])
    v_rtn = a_c[:, np.newaxis] * np.column_stack([vx, vy, vz])

    return x_rtn, v_rtn


def oe2los(chief_oe: np.ndarray, deputy_oe: np.ndarray, anomaly_flag: int,
           mu: float, chief_q_bi: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    # Calculate line of sight (LOS) from orbital elements for angles-only navigation

    Parameters
    ----------
    chief_oe : np.ndarray
        absolute orbital elements of chief, n x 6 matrix
        [a, e, i, RAAN, w, f (or M)] (m or km, -, rad, rad, rad, rad)
    deputy_oe : np.ndarray
        absolute orbital elements of deputy, n x 6 matrix
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly
    mu : float
        gravitational constant
    chief_q_bi : np.ndarray, optional
        quaternion from inertial to body-fixed frame of chief, 1 x 4
        If not provided, LOS is calculated in RTN frame

    Returns
    -------
    azi : np.ndarray
        angle from R-axis in R-T plane, n x 1, rad
    ele : np.ndarray
        angle from R-T plane toward N direction, n x 1, rad

    Notes
    -----
    重力定数と位置・速度は単位を合わせること

    References
    ----------
    Sullivan, Joshua, Generalized Angles-Only Navigation Architecture for
    Autonomous Distributed Space Systems, Journal of Guidance, Control, and Dynamics.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    roe2los_approx
    """
    from ..orbit import oe2rv, true_anomaly
    from ..attitude import zxz2q, q_rotation

    assert anomaly_flag in [0, 1], "anomaly_flag must be 0 or 1"

    chief_oe = np.atleast_2d(chief_oe)
    deputy_oe = np.atleast_2d(deputy_oe)

    # Calculate relative distance from absolute orbital elements
    r_c, _ = oe2rv(chief_oe, anomaly_flag, mu)
    r_d, _ = oe2rv(deputy_oe, anomaly_flag, mu)

    if chief_q_bi is None:
        # LOS is calculated in RTN frame
        if anomaly_flag == 1:
            f_c = chief_oe[:, 5]
        else:
            f_c, _ = true_anomaly(chief_oe[:, 0], chief_oe[:, 1], chief_oe[:, 5])

        f_c = np.mod(f_c, 2 * np.pi)
        inc_c = chief_oe[:, 2]
        raan_c = np.mod(chief_oe[:, 3], 2 * np.pi)
        u_c = np.mod(chief_oe[:, 4] + f_c, 2 * np.pi)

        q_ijk2rtn = zxz2q(raan_c, inc_c, u_c, scalar=4)
        rel = q_rotation(r_d - r_c, q_ijk2rtn, scalar=4)
    else:
        # LOS is calculated in body-fixed frame of chief
        rel = q_rotation(r_d - r_c, chief_q_bi, scalar=4)

    # LOS angles
    azi = np.arctan2(rel[:, 1], rel[:, 0])
    ele = np.arcsin(rel[:, 2] / np.linalg.norm(rel, axis=1))

    return azi, ele


def roe2deputy_oe(roe: np.ndarray, chief_oe: np.ndarray, anomaly_flag: int) -> np.ndarray:
    """
    # Convert ROE back to deputy absolute orbital elements

    Parameters
    ----------
    roe : np.ndarray
        relative orbital elements, n x 6
    chief_oe : np.ndarray
        chief absolute orbital elements, n x 6
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly

    Returns
    -------
    deputy_oe : np.ndarray
        deputy absolute orbital elements, n x 6
        (raan, w and the anomaly are wrapped to [0, 2pi))

    Revisions
    ---------
    20211027  y.yoshimura
    20261001  y.yoshimura, equatorial chief handled row by row (raan = chief raan)
    """
    from ..orbit.kepler import true_anomaly as calc_true_anomaly

    assert anomaly_flag in [0, 1], "anomaly_flag must be 0 or 1"

    roe = np.atleast_2d(roe)
    chief_oe = np.atleast_2d(chief_oe)

    a_c = chief_oe[:, 0]
    e_c = chief_oe[:, 1]
    inc_c = chief_oe[:, 2]
    raan_c = chief_oe[:, 3]
    w_c = chief_oe[:, 4]

    if anomaly_flag == 1:
        m_c = mean_anomaly(e_c, chief_oe[:, 5])
    else:
        m_c = chief_oe[:, 5]

    ex_c = e_c * np.cos(w_c)
    ey_c = e_c * np.sin(w_c)
    u_c = np.mod(w_c + m_c, 2 * np.pi)

    # Recover deputy parameters
    a_d = a_c * (1 + roe[:, 0])

    # Equatorial chief (row by row): raan is kept at the chief's value
    equatorial = inc_c < np.finfo(float).eps
    d_raan = np.where(equatorial, 0.0, roe[:, 5] / np.where(equatorial, 1.0, np.sin(inc_c)))
    raan_d = np.mod(raan_c + d_raan, 2 * np.pi)

    d_u = roe[:, 1] - d_raan * np.cos(inc_c)
    u_d = u_c + d_u

    ex_d = ex_c + roe[:, 2]
    ey_d = ey_c + roe[:, 3]

    e_d = np.sqrt(ex_d ** 2 + ey_d ** 2)
    w_d = np.mod(np.arctan2(ey_d, ex_d), 2 * np.pi)

    inc_d = inc_c + roe[:, 4]

    m_d = np.mod(u_d - w_d, 2 * np.pi)

    if anomaly_flag == 1:
        f_d, _ = calc_true_anomaly(a_d, e_d, m_d)
        anomaly_d = f_d
    else:
        anomaly_d = m_d

    deputy_oe = np.column_stack([a_d, e_d, inc_d, raan_d, w_d, anomaly_d])

    return deputy_oe


def roe2mapped_los(roe: np.ndarray, chief_oe: np.ndarray, anomaly_flag: int,
                   mu: float, chief_q_bo: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    # Calculate approximated line of sight (LOS) from relative orbital elements (ROE)

    Parameters
    ----------
    roe : np.ndarray
        relative orbital elements (ROE), n x 6 matrix
    chief_oe : np.ndarray
        absolute orbital elements of chief, n x 6 matrix
        [a, e, i, RAAN, w, f (or M)] (m or km, -, rad, rad, rad, rad)
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly
    mu : float
        gravitational constant (unit must be unified with position)
    chief_q_bo : np.ndarray, optional
        quaternion from RTN (o-frame) to chief's body-fixed frame, 1 x 4
        If not provided, LOS is calculated in RTN frame

    Returns
    -------
    azi : np.ndarray
        angle from -T axis toward R axis (in R-T plane), n x 1, rad
    ele : np.ndarray
        angle from R-T plane toward N direction, n x 1, rad

    Notes
    -----
    LOS calculation using 1st order ROE mapping to RTN frame.
    重力定数とsemi-major axisの単位を合わせること

    References
    ----------
    Di Mauro, G. 2019 Minimum-Fuel Control Strategy for Spacecraft Formation
    Reconfiguration via Finite-Time Maneuvers Journal of Guidance, Control, and Dynamics.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    oe2roe
    """
    from ..attitude import q_rotation

    # Mapping from ROE to RTN
    rel, _ = roe2rtn(roe, chief_oe, anomaly_flag, mu)  # n x 3

    if chief_q_bo is not None:
        # Transform to body-fixed frame of chief
        rel = q_rotation(rel, chief_q_bo, scalar=4)

    # LOS angles
    azi = np.arctan2(rel[:, 1], rel[:, 0])  # angle in R-T plane
    ele = np.arcsin(rel[:, 2] / np.linalg.norm(rel, axis=1))  # angle from R-T plane toward N

    return azi, ele


def calc_rel_pos_vel_atti(chief_oe: np.ndarray, deputy_oe: np.ndarray,
                          anomaly_flag: int, mu: float,
                          chief_q: np.ndarray = None, chief_n: float = None,
                          deputy_q: np.ndarray = None, deputy_w: np.ndarray = None
                          ) -> dict:
    """
    # Calculate relative position, velocity and attitude between chief and deputy

    Parameters
    ----------
    chief_oe : np.ndarray
        chief orbital elements, n x 6 matrix [a, e, i, RAAN, w, f/M]
    deputy_oe : np.ndarray
        deputy orbital elements, n x 6 matrix [a, e, i, RAAN, w, f/M]
    anomaly_flag : int
        1 = true anomaly, 0 = mean anomaly
    mu : float
        gravitational constant
    chief_q : np.ndarray, optional
        chief attitude quaternion (inertial to body), n x 4
    chief_n : float, optional
        chief mean motion, rad/s (needed if chief_q is provided)
    deputy_q : np.ndarray, optional
        deputy attitude quaternion (inertial to body), n x 4
    deputy_w : np.ndarray, optional
        deputy angular rate in body frame, rad/s, n x 3

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'chief_r_i': chief position in inertial frame, km, n x 3
        - 'chief_v_i': chief velocity in inertial frame, km/s, n x 3
        - 'deputy_r_i': deputy position in inertial frame, km, n x 3
        - 'deputy_v_i': deputy velocity in inertial frame, km/s, n x 3
        - 'rel_r_nonlin_i': nonlinear relative position in inertial frame, km, n x 3
        - 'rel_r_nonlin_rtn': nonlinear relative position in RTN frame, km, n x 3
        - 'rel_v_nonlin_i': nonlinear relative velocity in inertial frame, km/s, n x 3
        - 'rel_v_nonlin_rtn': nonlinear relative velocity in RTN frame, km/s, n x 3
        - 'roe': relative orbital elements, n x 6
        - 'rel_r_mapped_rtn': mapped relative position in RTN frame, km, n x 3
        - 'chief_q_oi': quaternion from inertial to RTN (o-frame), n x 4 (if chief_q provided)
        - 'chief_q_bo': quaternion from RTN to chief body, n x 4 (if chief_q provided)
        - 'deputy_q_bo': quaternion from RTN to deputy body, n x 4 (if deputy_q provided)
        - 'deputy_w_bo': deputy angular rate w.r.t. orbital frame, rad/s, n x 3 (if deputy_w provided)

    Notes
    -----
    Computes both nonlinear (from absolute OE) and linearized (from ROE) relative motion.

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura

    See also
    --------
    oe2roe, roe2rtn, oe2rv
    """
    from ..orbit import oe2rv
    from ..attitude import triad, dcm2q, q_mult, q_inv, q_rotation
    from ..math_utils import norm_row

    chief_oe = np.atleast_2d(chief_oe)
    deputy_oe = np.atleast_2d(deputy_oe)
    n = chief_oe.shape[0]

    # Absolute position and velocity in inertial frame
    chief_r_i, chief_v_i = oe2rv(chief_oe, anomaly_flag, mu)
    deputy_r_i, deputy_v_i = oe2rv(deputy_oe, anomaly_flag, mu)

    # Orbital angular momentum and RTN frame
    h_vec = np.cross(chief_r_i, chief_v_i)  # n x 3

    # DCM from inertial to RTN (o-frame) using TRIAD
    r_hat = norm_row(chief_r_i)
    h_hat = norm_row(h_vec)
    ref1 = np.tile([1, 0, 0], (n, 1))
    ref2 = np.tile([0, 0, 1], (n, 1))

    R_oi = triad(r_hat, h_hat, ref1, ref2)  # 3 x 3 x n

    result = {
        'chief_r_i': chief_r_i,
        'chief_v_i': chief_v_i,
        'deputy_r_i': deputy_r_i,
        'deputy_v_i': deputy_v_i,
    }

    # Quaternion from inertial to RTN frame
    chief_q_oi = np.zeros((n, 4))
    for i in range(n):
        chief_q_oi[i, :] = dcm2q(R_oi[:, :, i], scalar=4)

    result['chief_q_oi'] = chief_q_oi

    # If attitude quaternions are provided
    if chief_q is not None:
        chief_q = np.atleast_2d(chief_q)
        chief_q_bo = np.zeros((n, 4))
        for i in range(n):
            chief_q_bo[i, :] = q_mult(chief_q[i, :], q_inv(chief_q_oi[i, :], scalar=4), scalar=4)
        result['chief_q_bo'] = chief_q_bo

    if deputy_q is not None:
        deputy_q = np.atleast_2d(deputy_q)
        deputy_q_bo = np.zeros((n, 4))
        for i in range(n):
            deputy_q_bo[i, :] = q_mult(deputy_q[i, :], q_inv(chief_q_oi[i, :], scalar=4), scalar=4)
        result['deputy_q_bo'] = deputy_q_bo

        if deputy_w is not None and chief_n is not None:
            deputy_w = np.atleast_2d(deputy_w)
            # Angular rate of o-frame w.r.t. inertial expressed in deputy body frame
            w_oi_body = q_rotation(np.tile([0, 0, chief_n], (n, 1)), deputy_q_bo, scalar=4)
            deputy_w_bo = deputy_w - w_oi_body
            result['deputy_w_bo'] = deputy_w_bo

    # Nonlinear relative motion
    rel_r_nonlin_i = deputy_r_i - chief_r_i
    rel_r_nonlin_rtn = q_rotation(rel_r_nonlin_i, chief_q_oi, scalar=4)

    rel_v_nonlin_i = deputy_v_i - chief_v_i
    rel_v_nonlin_rtn = q_rotation(rel_v_nonlin_i, chief_q_oi, scalar=4)

    # Account for rotating frame
    if chief_n is not None:
        w_rtn = np.tile([0, 0, chief_n], (n, 1))
        rel_v_nonlin_rtn = rel_v_nonlin_rtn - np.cross(w_rtn, rel_r_nonlin_i)

    result['rel_r_nonlin_i'] = rel_r_nonlin_i
    result['rel_r_nonlin_rtn'] = rel_r_nonlin_rtn
    result['rel_v_nonlin_i'] = rel_v_nonlin_i
    result['rel_v_nonlin_rtn'] = rel_v_nonlin_rtn

    # ROE
    roe = oe2roe(chief_oe, deputy_oe, anomaly_flag)
    roe[:, 1] = wrap_pi(roe[:, 1])
    result['roe'] = roe

    # Mapped relative position from ROE
    rel_r_mapped_rtn, _ = roe2rtn(roe, chief_oe, anomaly_flag, mu)
    result['rel_r_mapped_rtn'] = rel_r_mapped_rtn

    return result


# %[appendix]{"version":"1.0"}
