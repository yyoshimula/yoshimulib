"""
environment - Space environment models (IGRF-12 magnetic field, Jaccia-Bowman atmosphere)

yoshimuLibrary environment module
Python conversion from yMATLAB/environment/
"""

from .magnetic import (
    igrf12,
    geodetic_igrf,
)

from .atmosphere import (
    JBData,
    jaccia_bowman,
    nrlmsise00_density,
    exponential_atmosphere,
    SCALE_HEIGHTS,
)

__all__ = [
    # magnetic
    'igrf12', 'geodetic_igrf',
    # atmosphere
    'JBData', 'jaccia_bowman', 'nrlmsise00_density',
    'exponential_atmosphere', 'SCALE_HEIGHTS',
]
