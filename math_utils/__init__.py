"""
math_utils - Mathematical utility functions

yoshimuLibrary math module
Python conversion from yMATLAB/math/
Note: Named math_utils to avoid conflict with Python's built-in math module
"""

from .core import (
    skew,
    wrap_pi,
    norm_row,
)

from .legendre import (
    associated_legendre,
)

__all__ = [
    # core
    'skew', 'wrap_pi', 'norm_row',
    # legendre
    'associated_legendre',
]
