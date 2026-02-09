"""
dual_quaternions - Dual quaternion operations for rigid body motion

yoshimuLibrary dual_quaternions module
Python conversion from yMATLAB/dualQuaternions/
"""

from .dq import (
    dq_mult,
    dq_conj,
    dq_inv,
    pos2dq,
    dq2pos,
    sclerp,
    cay,
    cay_inv,
    ctrl_dq,
    dq_cay,
    dq_mult_mat,
)

__all__ = [
    'dq_mult', 'dq_conj', 'dq_inv',
    'pos2dq', 'dq2pos',
    'sclerp',
    'cay', 'cay_inv',
    'ctrl_dq', 'dq_cay', 'dq_mult_mat',
]
