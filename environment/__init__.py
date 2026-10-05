"""
environment - Space environment models (IGRF-12 magnetic field, Jaccia-Bowman and
Jacchia-Roberts 1971 atmosphere, space weather indices, disturbance accelerations)

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

from .jacchia_roberts import (
    jr1971,
)

from .space_weather import (
    load_space_weather,
    lookup_solar_geo_index,
)

from .disturbances import (
    disturbances,
)

__all__ = [
    # magnetic
    'igrf12', 'geodetic_igrf',
    # atmosphere
    'JBData', 'jaccia_bowman', 'nrlmsise00_density',
    'exponential_atmosphere', 'SCALE_HEIGHTS',
    'jr1971',
    # space weather
    'load_space_weather', 'lookup_solar_geo_index',
    # disturbances
    'disturbances',
]
