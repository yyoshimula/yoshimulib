"""
time_utils - Time system conversion functions

yoshimuLibrary time module
Python conversion from yMATLAB/time/
Note: Named time_utils to avoid conflict with Python's built-in time module
"""

from .julian import (
    jd2gc,
    jd2jdt,
    jd2mjd,
    mjd2jd,
    doy2gc,
)

from .timescales import (
    leap_s,
    dat,
    utc2tt,
    ut2tt,
)

__all__ = [
    # julian
    'jd2gc', 'jd2jdt', 'jd2mjd', 'mjd2jd', 'doy2gc',
    # timescales
    'leap_s', 'dat', 'utc2tt', 'ut2tt',
]
