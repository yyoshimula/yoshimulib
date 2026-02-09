"""
lightcurves - Synthetic light curve generation using BRDF models

yoshimuLibrary lightcurves module
Python conversion from yMATLAB/lightcurves/
"""

from .lc_models import (
    lc,
    lc_simple,
    lc_as,
    lc_ct,
    mag,
    mag_inv,
)

__all__ = ['lc', 'lc_simple', 'lc_as', 'lc_ct', 'mag', 'mag_inv']
