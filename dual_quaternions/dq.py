"""
Dual Quaternion operations
Python conversion from yMATLAB/dualQuaternions/
"""

import numpy as np
from ..attitude import q_mult, q_conj, q_inv


def dq_mult(scalar: int, definition: int, dq: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """
    # dual quaternion multiplication

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T = [cos(theta/2), e^T*sin(theta/2)]^T
        scalar == 4: q = [q1, q2, q3, q4]^T = [e^T*sin(theta/2), cos(theta/2)]^T
    definition : int
        specifies the definition of the quaternion multiplication
        definition == 0: q (x) p (standard)
        definition == 1: q (*) p (alternative)
    dq : np.ndarray
        dual quaternion [real, dual], n x 8
    dp : np.ndarray
        dual quaternion [real, dual], n x 8

    Returns
    -------
    dq_out : np.ndarray
        result dual quaternion [real, dual], n x 8

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20231219  y.yoshimura

    See also
    --------
    q_mult
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"
    assert definition in [0, 1], "definition must be 0 or 1"

    dq = np.atleast_2d(dq)
    dp = np.atleast_2d(dp)

    dq_real = dq[:, :4]
    dq_dual = dq[:, 4:]

    dp_real = dp[:, :4]
    dp_dual = dp[:, 4:]

    real_part = q_mult(dq_real, dp_real, scalar=scalar, definition=definition)
    dual_part = (q_mult(dq_real, dp_dual, scalar=scalar, definition=definition) +
                 q_mult(dq_dual, dp_real, scalar=scalar, definition=definition))

    dq_out = np.hstack([real_part, dual_part])

    return dq_out


# %[appendix]{"version":"1.0"}


def dq_conj(scalar: int, dq: np.ndarray) -> np.ndarray:
    """
    # Dual quaternion conjugate

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T = [cos(theta/2), e^T*sin(theta/2)]^T
        scalar == 4: q = [q1, q2, q3, q4]^T = [e^T*sin(theta/2), cos(theta/2)]^T
    dq : np.ndarray
        dual quaternion [real, dual], n x 8

    Returns
    -------
    inv_dq : np.ndarray
        dual quaternion conjugate [real, dual], n x 8

    Notes
    -----
    NA

    References
    ----------
    Sveier, A., & Egeland, O. (2020). Dual Quaternion Particle Filtering for
    Pose Estimation. IEEE Transactions on Control Systems Technology, 1-14.

    Revisions
    ---------
    20220526  y.yoshimura

    See also
    --------
    q_mult
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"

    dq = np.atleast_2d(dq)

    if scalar == 0:
        inv_dq = np.hstack([dq[:, 0:1], -dq[:, 1:4], dq[:, 4:5], -dq[:, 5:8]])
    else:
        inv_dq = np.hstack([-dq[:, 0:3], dq[:, 3:4], -dq[:, 4:7], dq[:, 7:8]])

    return inv_dq


# %[appendix]{"version":"1.0"}


def dq_inv(scalar: int, dq: np.ndarray) -> np.ndarray:
    """
    # Inverse dual quaternions

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    dq : np.ndarray
        dual quaternion [real, dual], n x 8

    Returns
    -------
    inv_dq : np.ndarray
        inverse dual quaternion [real, dual], n x 8

    Notes
    -----
    NA

    References
    ----------
    Sveier, A., & Egeland, O. (2020). Dual Quaternion Particle Filtering for
    Pose Estimation. IEEE Transactions on Control Systems Technology, 1-14.

    Revisions
    ---------
    20220526  y.yoshimura

    See also
    --------
    q_mult, dq_conj
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"

    dq = np.atleast_2d(dq)

    dqr = dq[:, :4]  # real part
    dqd = dq[:, 4:]  # dual part

    inv_dqr = q_inv(dqr, scalar=scalar)
    inv_dqd = -q_mult(q_mult(inv_dqr, dqd, scalar=scalar, definition=0),
                      inv_dqr, scalar=scalar, definition=0)

    inv_dq = np.hstack([inv_dqr, inv_dqd])

    return inv_dq


# %[appendix]{"version":"1.0"}


def pos2dq(inertial: int, scalar: int, r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    # transforming position and quaternion to dual quaternion

    Parameters
    ----------
    inertial : int
        0: body-fixed frame, 1: position vector is expressed with inertial frame
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    r : np.ndarray
        position vector, n x 3
    q : np.ndarray
        quaternion, n x 4

    Returns
    -------
    dq : np.ndarray
        dual quaternion [real, dual], n x 8

    Notes
    -----
    NA

    References
    ----------
    Sveier, A., & Egeland, O. (2020). Dual Quaternion Particle Filtering for
    Pose Estimation. IEEE Transactions on Control Systems Technology, 1-14.

    Revisions
    ---------
    20220526  y.yoshimura

    See also
    --------
    q_mult
    """
    assert inertial in [0, 1], "inertial must be 0 or 1"
    assert scalar in [0, 4], "scalar must be 0 or 4"

    r = np.atleast_2d(r)
    q = np.atleast_2d(q)
    n = r.shape[0]

    # Real part
    dq_real = q

    # Dual part
    if scalar == 0:
        pos = np.hstack([np.zeros((n, 1)), r])
    else:
        pos = np.hstack([r, np.zeros((n, 1))])

    if inertial == 0:
        dq_dual = 0.5 * q_mult(pos, q, scalar=scalar, definition=1)
    else:
        dq_dual = 0.5 * q_mult(q, pos, scalar=scalar, definition=1)

    dq = np.hstack([dq_real, dq_dual])

    return dq


# %[appendix]{"version":"1.0"}


def dq2pos(inertial: int, scalar: int, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    # transforming dual quaternion to position and quaternion

    Parameters
    ----------
    inertial : int
        1: position vector is expressed with inertial frame, 0: body-fixed frame
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    dq : np.ndarray
        dual quaternion [real, dual], n x 8

    Returns
    -------
    r : np.ndarray
        position vector, n x 3
    q : np.ndarray
        quaternion, n x 4

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    20231219  y.yoshimura

    See also
    --------
    dq_mult, pos2dq
    """
    assert inertial in [0, 1], "inertial must be 0 or 1"
    assert scalar in [0, 4], "scalar must be 0 or 4"

    dq = np.atleast_2d(dq)

    # Quaternion = real part
    q = dq[:, :4]

    # Dual part
    if inertial == 0:
        tmp_r = q_mult(dq[:, 4:], q_inv(q, scalar=scalar), scalar=scalar, definition=1)
    else:
        tmp_r = q_mult(q_inv(q, scalar=scalar), dq[:, 4:], scalar=scalar, definition=1)

    tmp_r = tmp_r * 2.0

    if scalar == 0:
        r = tmp_r[:, 1:4]
    else:
        r = tmp_r[:, :3]

    return r, q


# %[appendix]{"version":"1.0"}


def sclerp(t: np.ndarray, scalar: int, dq1: np.ndarray, dq2: np.ndarray) -> np.ndarray:
    """
    # Screw linear interpolation of dual quaternions (sclerp)

    Parameters
    ----------
    t : np.ndarray
        normalized time, i.e., 0 < t <= 1
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    dq1 : np.ndarray
        starting dual quaternion, 1 x 8
    dq2 : np.ndarray
        ending dual quaternion, 1 x 8

    Returns
    -------
    dqt : np.ndarray
        interpolated dual quaternions, n x 8

    Notes
    -----
    NA

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    dq_mult, dq_inv
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"

    t = np.atleast_1d(t).flatten()
    dq1 = np.atleast_2d(dq1)
    dq2 = np.atleast_2d(dq2)
    n_t = len(t)

    # Rotation
    dq_tmp = dq_mult(scalar, 0, dq_inv(scalar, dq1), dq2).flatten()

    if scalar == 0:
        e_theta = 2 * np.arccos(np.clip(dq_tmp[0], -1, 1))
        e_axis = dq_tmp[1:4]
        d = dq_tmp[4]
        dq_vec = dq_tmp[5:8]
    else:
        e_theta = 2 * np.arccos(np.clip(dq_tmp[3], -1, 1))
        e_axis = dq_tmp[0:3]
        d = dq_tmp[7]
        dq_vec = dq_tmp[4:7]

    sin_half_theta = np.sin(e_theta / 2)
    cos_half_theta = np.cos(e_theta / 2)

    # Zero division check
    if np.abs(sin_half_theta) < np.finfo(float).eps:
        return np.tile(dq1, (n_t, 1))

    e_axis = e_axis / sin_half_theta
    d = -2 * d / sin_half_theta
    p = (dq_vec - d / 2 * cos_half_theta * e_axis) / sin_half_theta

    # Vectorized trigonometric calculations
    t_theta_half = t * e_theta / 2
    cos_t_theta_half = np.cos(t_theta_half)
    sin_t_theta_half = np.sin(t_theta_half)

    t_d_half = t * d / 2
    e_axis_rep = np.tile(e_axis, (n_t, 1))
    p_rep = np.tile(p, (n_t, 1))

    neg_t_d_half_sin = -t_d_half * sin_t_theta_half
    t_d_half_cos_e_axis = t_d_half[:, np.newaxis] * cos_t_theta_half[:, np.newaxis] * e_axis_rep

    # Build interpolated dual quaternion
    if scalar == 0:
        dq_interp = np.hstack([
            cos_t_theta_half[:, np.newaxis],
            e_axis_rep * sin_t_theta_half[:, np.newaxis],
            neg_t_d_half_sin[:, np.newaxis],
            p_rep * sin_t_theta_half[:, np.newaxis] + t_d_half_cos_e_axis
        ])
    else:
        dq_interp = np.hstack([
            e_axis_rep * sin_t_theta_half[:, np.newaxis],
            cos_t_theta_half[:, np.newaxis],
            p_rep * sin_t_theta_half[:, np.newaxis] + t_d_half_cos_e_axis,
            neg_t_d_half_sin[:, np.newaxis]
        ])

    # Final multiplication
    dq1_rep = np.tile(dq1, (n_t, 1))
    dqt = dq_mult(scalar, 0, dq1_rep, dq_interp)

    return dqt


# %[appendix]{"version":"1.0"}


def cay(scalar: int, u: np.ndarray) -> np.ndarray:
    """
    # Cayley transformation

    cay(u) = (1 + u) (*) (1 - u)^{-1}

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    u : np.ndarray
        arbitrary vector, n x 3

    Returns
    -------
    q : np.ndarray
        quaternion, n x 4

    Notes
    -----
    In the definition, 1-u means quaternion [u^T, 1]^T.

    References
    ----------
    Sveier, A., & Egeland, O. (2020). Dual Quaternion Particle Filtering for
    Pose Estimation. IEEE Transactions on Control Systems Technology, 1-14.

    Revisions
    ---------
    20220526  y.yoshimura

    See also
    --------
    q_mult
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"

    u = np.atleast_2d(u)
    n = u.shape[0]

    # u(4) = scalar part as calculation basis
    if scalar == 0:
        u_q1 = np.hstack([np.ones((n, 1)), u])
    else:
        u_q1 = np.hstack([u, np.ones((n, 1))])

    u_q2 = q_conj(u_q1, scalar=4)

    # Calculate using scalar=4 convention
    q = q_mult(u_q1, q_inv(u_q2, scalar=scalar), scalar=4, definition=0)

    if scalar == 0:
        q = np.hstack([q[:, 3:4], q[:, :3]])

    return q


# %[appendix]{"version":"1.0"}


def cay_inv(scalar: int, q: np.ndarray) -> np.ndarray:
    """
    # Inverse Cayley transformation

    cay^{-1}(q) = (q - 1) (*) (q + 1)^{-1}

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    q : np.ndarray
        quaternion, n x 4

    Returns
    -------
    u : np.ndarray
        arbitrary vector, n x 3

    Notes
    -----
    In the definition, 1 means unit quaternion [0, 0, 0, 1]^T.

    References
    ----------
    Sveier, A., & Egeland, O. (2020). Dual Quaternion Particle Filtering for
    Pose Estimation. IEEE Transactions on Control Systems Technology, 1-14.

    Revisions
    ---------
    20220526  y.yoshimura

    See also
    --------
    q_mult
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"

    q = np.atleast_2d(q)
    n = q.shape[0]

    # q4 = scalar part as calculation basis
    if scalar == 0:
        q = np.hstack([q[:, 1:4], q[:, 0:1]])

    q_unit = np.hstack([np.zeros((n, 3)), np.ones((n, 1))])

    tmp = q_mult(q - q_unit, q_inv(q + q_unit, scalar=4), scalar=4, definition=0)

    u = tmp[:, :3]

    return u


# %[appendix]{"version":"1.0"}


def ctrl_dq(dtp: float, dq1: np.ndarray, dq2: np.ndarray,
            w1: np.ndarray, v1: np.ndarray,
            w2: np.ndarray, v2: np.ndarray,
            scalar: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    # Control dual quaternions for C^1 sclerp

    Parameters
    ----------
    dtp : float
        time step, s
    dq1 : np.ndarray
        dual quaternion at start, 1 x 8
    dq2 : np.ndarray
        dual quaternion at end, 1 x 8
    w1 : np.ndarray
        angular rate at start, 1 x 3
    v1 : np.ndarray
        translational velocity at start, 1 x 3
    w2 : np.ndarray
        angular rate at end, 1 x 3
    v2 : np.ndarray
        translational velocity at end, 1 x 3
    scalar : int, optional
        quaternion convention (default: 4)

    Returns
    -------
    dqa : np.ndarray
        control dual quaternion a, 1 x 8
    dqb : np.ndarray
        control dual quaternion b, 1 x 8

    Notes
    -----
    NA

    References
    ----------
    Allmendinger, F., Charaf Eddine, S., & Corves, B. (2018).
    Coordinate-invariant rigid-body interpolation on a parametric C1
    dual quaternion curve. Mechanism and Machine Theory, 121, 731-744.

    Revisions
    ---------
    20210310  y.yoshimura

    See also
    --------
    sclerp, dq_mult, dq_conj
    """
    w1 = np.asarray(w1).flatten()
    v1 = np.asarray(v1).flatten()
    w2 = np.asarray(w2).flatten()
    v2 = np.asarray(v2).flatten()
    dq1 = np.atleast_2d(dq1)
    dq2 = np.atleast_2d(dq2)

    # For dqa
    tmp = np.concatenate([dtp / 6.0 * w1, dtp / 6.0 * v1])
    e_theta = 2.0 * np.linalg.norm(tmp[:3])

    if e_theta > 1e-10:
        e_axis = tmp[:3] / (e_theta / 2)
    else:
        e_axis = np.array([0, 0, 1])

    dqtmp_r = np.concatenate([e_axis * np.sin(e_theta / 2), [np.cos(e_theta / 2)]])
    dqtmp_d = 0.5 * q_mult(np.concatenate([tmp[3:6], [0]]), dqtmp_r, scalar=4, definition=0)
    dq_tmp = np.concatenate([dqtmp_r, dqtmp_d]).reshape(1, -1)
    dqa = dq_mult(scalar, 0, dq1, dq_tmp)

    # For dqb
    tmp = np.concatenate([-dtp / 6.0 * w2, -dtp / 6.0 * v2])
    e_theta = 2.0 * np.linalg.norm(tmp[:3])

    if e_theta > 1e-10:
        e_axis = tmp[:3] / (e_theta / 2)
    else:
        e_axis = np.array([0, 0, 1])

    dqtmp_r = np.concatenate([e_axis * np.sin(e_theta / 2), [np.cos(e_theta / 2)]])
    dqtmp_d = 0.5 * q_mult(np.concatenate([tmp[3:6], [0]]), dqtmp_r, scalar=4, definition=0)
    dq_tmp = np.concatenate([dqtmp_r, dqtmp_d]).reshape(1, -1)
    dqb = dq_mult(scalar, 0, dq2, dq_tmp)

    return dqa.flatten(), dqb.flatten()


# %[appendix]{"version":"1.0"}


def dq_cay(scalar: int, dv: np.ndarray) -> np.ndarray:
    """
    # Dual Cayley transformation

    cay(u_tilde) = (1 + u_tilde) (*) (1 - u_tilde)^{-1}

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    dv : np.ndarray
        arbitrary dual vector (scalar part is 0), n x 8

    Returns
    -------
    dq : np.ndarray
        dual quaternion, n x 8

    Notes
    -----
    The 1+ in the formula means setting the scalar part of the real
    quaternion to 1, i.e., adding 1 + eps*0.

    References
    ----------
    Sveier, A., & Egeland, O. (2020). Dual Quaternion Particle Filtering
    for Pose Estimation. IEEE Transactions on Control Systems Technology, 1-14.

    Revisions
    ---------
    20240111  y.yoshimura

    See also
    --------
    cay
    """
    assert scalar in [0, 4], "scalar must be 0 or 4"

    dv = np.atleast_2d(dv)
    n = dv.shape[0]

    # Real part: scalar part set to 1
    if scalar == 0:
        dv_real = np.hstack([np.ones((n, 1)), dv[:, 1:4]])
        tmp = dv[:, 1:4]  # for real part calculation
    else:
        dv_real = np.hstack([dv[:, 0:3], np.ones((n, 1))])
        tmp = dv[:, 0:3]

    dv_dual = dv[:, 4:8]

    # Cayley transformation
    # Real part
    dq_real = cay(scalar, tmp)

    # Dual part
    dq_dual = 2 * q_mult(q_mult(dv_real, dv_dual, scalar=scalar, definition=0),
                         dv_real, scalar=scalar, definition=0)
    dq_dual = dq_dual / (np.linalg.norm(dv_real, axis=1, keepdims=True) ** 2) ** 2

    dq = np.hstack([dq_real, dq_dual])

    return dq


# %[appendix]{"version":"1.0"}


def dq_mult_mat(scalar: int, definition: int, dq: np.ndarray) -> np.ndarray:
    """
    # Dual quaternion multiplication matrix

    Parameters
    ----------
    scalar : int
        specify the definition of the quaternion
        scalar == 0: q = [q0, q1, q2, q3]^T
        scalar == 4: q = [q1, q2, q3, q4]^T
    definition : int
        specifies the definition of the quaternion multiplication
        definition == 0: q (x) p (standard)
        definition == 1: q (*) p (alternative)
    dq : np.ndarray
        dual quaternion [real, dual], 1 x 8

    Returns
    -------
    dq_M : np.ndarray
        dual quaternion multiplication matrix, 8 x 8

    Notes
    -----
    For dual quaternion q_tilde:
    q_tilde (*) p_tilde = [q_r(*)  0_{4x4}]
                          [q_d(*)  q_r(*) ]

    where q (*) p is the standard quaternion multiplication.

    References
    ----------
    NA

    Revisions
    ---------
    20240109  y.yoshimura

    See also
    --------
    q_mult, q_mult_mat, dq_mult
    """
    from ..attitude import q_mult_mat

    assert scalar in [0, 4], "scalar must be 0 or 4"
    assert definition in [0, 1], "definition must be 0 or 1"

    dq = np.atleast_2d(dq).flatten()

    dq_real = dq[:4]
    dq_dual = dq[4:]

    dq_M = np.zeros((8, 8))
    dq_M[:4, :4] = q_mult_mat(dq_real, scalar=scalar, definition=definition)
    dq_M[4:, :4] = q_mult_mat(dq_dual, scalar=scalar, definition=definition)
    dq_M[4:, 4:] = q_mult_mat(dq_real, scalar=scalar, definition=definition)

    return dq_M


# %[appendix]{"version":"1.0"}
