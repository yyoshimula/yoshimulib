"""
gpr - Gaussian Process Regression

yoshimuLibrary gpr module
Python conversion from yMATLAB/gpr/
"""

from .gpr import (
    kernel_gauss,
    kernel_gauss_mat,
    gpr_mean,
    gpr_cov,
)

__all__ = [
    'kernel_gauss', 'kernel_gauss_mat', 'gpr_mean', 'gpr_cov',
]
