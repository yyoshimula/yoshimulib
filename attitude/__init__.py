"""
attitude - Attitude dynamics and quaternion operations library

yoshimuLibrary attitude module
Python conversion from yMATLAB/attitude/
"""

from .dcm import (
    dcm1axis,
    dcm1axis_x,
    dcm1axis_y,
    dcm1axis_z,
    zyx2dcm,
    zxz2dcm,
    zyz2dcm,
)

from .quaternion import (
    q_conj,
    q_inv,
    q_mult,
    q2dcm,
    dcm2q,
    dcm2q_continuous,
)

from .euler import (
    zyx2q,
    zxz2q,
    zyz2q,
    q2zyx,
    q2zyz,
    euler_dcm,
    generate_euler_angle_kinematics,
)

from .conversions import (
    q1axis,
    q_axis_angle,
    q2rot_vec,
    rot_vec2q,
    q2rodrigues,
    rodrigues2q,
    q2grp,
    grp2q,
)

from .operations import (
    slerp,
    q_ave,
    mean_angle,
    q_err,
    q_rotation,
    triad,
)

from .kinematics import (
    q_kine,
    q_prop_mat,
    q_mult_mat,
    rot_period,
)

__all__ = [
    # dcm
    'dcm1axis', 'dcm1axis_x', 'dcm1axis_y', 'dcm1axis_z',
    'zyx2dcm', 'zxz2dcm', 'zyz2dcm',
    # quaternion
    'q_conj', 'q_inv', 'q_mult', 'q2dcm', 'dcm2q', 'dcm2q_continuous',
    # euler
    'zyx2q', 'zxz2q', 'zyz2q', 'q2zyx', 'q2zyz',
    'euler_dcm', 'generate_euler_angle_kinematics',
    # conversions
    'q1axis', 'q_axis_angle', 'q2rot_vec', 'rot_vec2q',
    'q2rodrigues', 'rodrigues2q', 'q2grp', 'grp2q',
    # operations
    'slerp', 'q_ave', 'mean_angle', 'q_err', 'q_rotation', 'triad',
    # kinematics
    'q_kine', 'q_prop_mat', 'q_mult_mat', 'rot_period',
]
