"""
Geometric Integration methods for attitude dynamics
Python conversion from yMATLAB/geometricIntegration/
"""

import numpy as np
from scipy.linalg import expm
from ..attitude import q_mult_mat


def butcher_table(n: int, method: str = 'CG') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Butcher table for geometric integration

    Parameters
    ----------
    n : int
        the order of geometric integration method (NOT the number of stages)
    method : str
        'RK' = Runge-Kutta method
        'CG' = Crouch-Grossman method

    Returns
    -------
    a : np.ndarray
        coefficient matrix
    b : np.ndarray
        coefficient vector
    c : np.ndarray
        coefficient vector

    Notes
    -----
    NA

    References
    ----------
    To be added

    Revisions
    ---------
    20240109  y.yoshimura y.yoshimula@gmail.com

    See also
    --------
    q_gi
    """
    if method == 'CG':
        # Crouch-Grossman method
        if n == 3:
            # Three-stage third-order Crouch-Grossman method (CG3)
            a = np.array([
                [0, 0, 0],
                [3/4, 0, 0],
                [119/216, 17/108, 0]
            ])
            b = np.array([13/51, -2/3, 24/17])
            c = np.array([0, 3/4, 17/24])
        elif n == 4:
            # Five-stage fourth-order Crouch-Grossman method (CG4)
            a = np.array([
                [0, 0, 0, 0, 0],
                [0.8177227988124852, 0, 0, 0, 0],
                [0.3199876375476427, 0.0659864263556022, 0, 0, 0],
                [0.9214417194464946, 0.4997857776773573, -1.0969984448371582, 0, 0],
                [0.3552358559023322, 0.2390958372307326, 1.3918565724203246, -1.1092979392113565, 0]
            ])
            b = np.array([
                0.1370831520630755,
                -0.0183698531564020,
                0.7397813985370780,
                -0.1907142565505889,
                0.3322195591068374
            ])
            c = np.array([
                0.0,
                0.8177227988124852,
                0.3859740639032449,
                0.3242290522866937,
                0.8768903263420429
            ])
        else:
            raise ValueError(f"No data for order {n} with CG method")

    elif method == 'RK':
        # Runge-Kutta method
        if n == 3:
            # Three-stage third-order Runge-Kutta method (RK3)
            a = np.array([
                [0, 0, 0],
                [1/2, 0, 0],
                [-1, 2, 0]
            ])
            b = np.array([1/6, 2/3, 1/6])
            c = np.array([0, 1/2, 1])
        elif n == 4:
            # Four-stage fourth-order Runge-Kutta method (RK4)
            a = np.array([
                [0, 0, 0, 0],
                [1/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 1, 0]
            ])
            b = np.array([1/6, 1/3, 1/3, 1/6])
            c = np.array([0, 1/2, 1/2, 1])
        else:
            raise ValueError(f"No data for order {n} with RK method")
    else:
        raise ValueError(f"Unknown method: {method}")

    return a, b, c


# %[appendix]{"version":"1.0"}


def q_exp(scalar: int, dt: float, w: np.ndarray) -> np.ndarray:
    """
    # Quaternion exponential map matrix

    Parameters
    ----------
    scalar : int
        quaternion scalar position (0 or 4)
    dt : float
        time step
    w : np.ndarray
        angular velocity vector, 3x1

    Returns
    -------
    q_mat : np.ndarray
        quaternion propagation matrix, 4x4

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
    q_gi
    """
    w = np.asarray(w).flatten()
    w_norm = np.linalg.norm(w)

    if w_norm < np.finfo(float).eps:
        return np.eye(4)

    q_mat = (np.eye(4) * np.cos(0.5 * dt * w_norm) +
             q_mult_mat(np.append(w, 0), scalar=scalar, definition=1) / w_norm * np.sin(0.5 * dt * w_norm))

    return q_mat


# %[appendix]{"version":"1.0"}


def q_gi(scalar: int, tspan: np.ndarray, q_ini: np.ndarray, w_ini: np.ndarray,
         n_gi: int, moi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    # Geometric integration of quaternion (mixed-scheme)

    Parameters
    ----------
    scalar : int
        quaternion scalar position (0 or 4)
    tspan : np.ndarray
        time span vector
    q_ini : np.ndarray
        initial quaternion, 4x1
    w_ini : np.ndarray
        initial angular velocity, 3x1
    n_gi : int
        order of geometric integration (3 or 4)
    moi : np.ndarray
        moment of inertia tensor, 3x3

    Returns
    -------
    q_out : np.ndarray
        quaternion history, n x 4
    w_out : np.ndarray
        angular velocity history, n x 3

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
    butcher_table, q_exp
    """
    a, b, _ = butcher_table(n_gi, 'CG')

    if n_gi == 3:
        n_stage = 3
    elif n_gi == 4:
        n_stage = 5
    else:
        raise ValueError(f"Unsupported order: {n_gi}")

    n_t = len(tspan)
    kw = np.zeros((n_stage, 3))
    K = np.zeros((4, 4, n_stage))
    wi = np.zeros((n_stage, 3))

    wi[0, :] = w_ini

    w_out = np.zeros((n_t, 3))
    q_out = np.zeros((n_t, 4))

    w_out[0, :] = wi[0, :]

    # q4 == scalar part for calculation
    if scalar == 1:
        q_out[0, :] = np.array([q_ini[1], q_ini[2], q_ini[3], q_ini[0]])
    else:
        q_out[0, :] = q_ini

    dt = np.diff(tspan)
    moi_inv = np.linalg.inv(moi)

    # Geometric integration
    for k in range(n_t - 1):
        # When i = 0
        wi[0, :] = w_out[k, :]
        tmp = -moi_inv @ np.cross(wi[0, :], moi @ wi[0, :])
        kw[0, :] = tmp
        K[:, :, 0] = 0.5 * q_mult_mat(np.append(wi[0, :], 0), scalar=4, definition=1)

        for i in range(1, n_stage):  # for each stage
            tmp_w = wi[0, :].copy()
            for j in range(i):
                tmp_w = tmp_w + a[i, j] * dt[k] * kw[j, :]
            wi[i, :] = tmp_w

            tmp = -moi_inv @ np.cross(wi[i, :], moi @ wi[i, :])
            kw[i, :] = tmp
            K[:, :, i] = 0.5 * q_mult_mat(np.append(wi[i, :], 0), scalar=4, definition=1)

        tmp_w = np.zeros(3)
        for i in range(n_stage):
            tmp_w = tmp_w + dt[k] * b[i] * kw[i, :]
        w_out[k + 1, :] = w_out[k, :] + tmp_w

        tmp_q = q_out[k, :].copy()
        for i in range(n_stage):
            # Use matrix exponent
            tmp_q = expm(dt[k] * b[i] * K[:, :, i]) @ tmp_q

        q_out[k + 1, :] = tmp_q

    return q_out, w_out


# %[appendix]{"version":"1.0"}


def axi_q_sol(t: np.ndarray, q_ini: np.ndarray, w_ini: np.ndarray,
              moi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    # Axisymmetric quaternion solution (analytical)

    Parameters
    ----------
    t : np.ndarray
        time vector
    q_ini : np.ndarray
        initial quaternion (scalar=4 convention), 4x1
    w_ini : np.ndarray
        initial angular velocity, 3x1
    moi : np.ndarray
        moment of inertia tensor (axisymmetric), 3x3

    Returns
    -------
    q_out : np.ndarray
        quaternion history, n x 4
    w_out : np.ndarray
        angular velocity history, n x 3

    Notes
    -----
    Assumes axisymmetric body (MOI(1,1) = MOI(2,2))

    References
    ----------
    Andrle, M. S. & Crassidis, J. L. Geometric Integration of Quaternions.
    J. Guid., Control, Dyn. 36, 1762-1767 (2013).

    Revisions
    ---------
    NA

    See also
    --------
    q_gi
    """
    from ..attitude import q_mult

    t = np.atleast_1d(t).flatten()
    w_ini = np.asarray(w_ini).flatten()
    q_ini = np.asarray(q_ini).flatten()

    wn = w_ini[2] * (moi[0, 0] - moi[2, 2]) / moi[0, 0]

    # Angular velocity output
    w_out = np.zeros((len(t), 3))
    w_out[:, 0] = w_ini[0] * np.cos(wn * t) + w_ini[1] * np.sin(wn * t)
    w_out[:, 1] = w_ini[1] * np.cos(wn * t) - w_ini[0] * np.sin(wn * t)
    w_out[:, 2] = w_ini[2] * np.ones(len(t))

    # Angular momentum direction
    H_ini = moi @ w_ini
    h0 = H_ini / np.linalg.norm(H_ini)

    wi = np.linalg.norm(H_ini) / moi[0, 0]

    alp = 0.5 * wn * t
    bet = 0.5 * wi * t

    # Quaternion computation
    y = np.column_stack([
        h0[0] * np.cos(alp) * np.sin(bet) + h0[1] * np.sin(alp) * np.sin(bet),
        h0[1] * np.cos(alp) * np.sin(bet) - h0[0] * np.sin(alp) * np.sin(bet),
        h0[2] * np.cos(alp) * np.sin(bet) + np.sin(alp) * np.cos(bet),
        np.cos(alp) * np.cos(bet) - h0[2] * np.sin(alp) * np.sin(bet)
    ])

    q_out = q_mult(y, q_ini, scalar=4, definition=1)

    return q_out, w_out


# %[appendix]{"version":"1.0"}


def euler_eom(t: float, x: np.ndarray, moi: np.ndarray,
              u: np.ndarray = None) -> np.ndarray:
    """
    # Equations of motion for rigid body (Euler's equations)

    Parameters
    ----------
    t : float
        time (not used, for ODE solver compatibility)
    x : np.ndarray
        state vector [q1, q2, q3, q4, wx, wy, wz], 7x1
    moi : np.ndarray
        moment of inertia tensor, 3x3
    u : np.ndarray, optional
        control torque input, 3x1 (default: zeros)

    Returns
    -------
    dxdt : np.ndarray
        state derivative [dq, dw], 7x1

    Notes
    -----
    Uses scalar=4 quaternion convention.
    State: [q1, q2, q3, q4, wx, wy, wz]
    where q = [q1, q2, q3, q4] with q4 as scalar part

    References
    ----------
    NA

    Revisions
    ---------
    20130212  yasuhiro yoshimura

    See also
    --------
    q_gi, axi_q_sol
    """
    from ..attitude import q_mult

    x = np.asarray(x).flatten()

    # State variables
    q = x[0:4]
    w_vec = x[4:7]

    # Control inputs
    if u is None:
        u = np.zeros(3)
    else:
        u = np.asarray(u).flatten()

    # Quaternion derivative: dq = 0.5 * q (*) [w; 0]
    w_quat = np.append(w_vec, 0)
    dq = 0.5 * q_mult(w_quat, q, scalar=4, definition=1)

    # Angular velocity derivative: dw = MOI^{-1} * (-w x (MOI * w) + u)
    moi_inv = np.linalg.inv(moi)
    dw = moi_inv @ (-np.cross(w_vec, moi @ w_vec) + u)

    dxdt = np.concatenate([dq, dw])

    return dxdt


# %[appendix]{"version":"1.0"}
