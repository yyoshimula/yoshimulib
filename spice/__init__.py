"""
spice - SPICE integration for ephemeris and reference frame computations

yoshimuLibrary spice module
Python conversion from yMATLAB/SPICE/

Note: For full SPICE functionality, requires SpiceyPy package:
  pip install spiceypy
"""

from .kernels import (
    load_spice_kernel,
    furnsh,
    kclear,
    ktotal,
    list_loaded_kernels,
)

__all__ = [
    'load_spice_kernel',
    'furnsh', 'kclear', 'ktotal',
    'list_loaded_kernels',
]
