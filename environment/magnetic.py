"""
Magnetic field models
Python conversion from yMATLAB/environment/
"""

import numpy as np
from typing import Optional


def igrf12(year: float, alt: float, lat: float, lon: float,
           model: str = 'ppigrf') -> np.ndarray:
    """
    # Calculate magnetic field with IGRF12 model

    Parameters
    ----------
    year : float
        decimal year (e.g., 2020.5)
    alt : float
        altitude above sea level, km
    lat : float
        north latitude, rad
    lon : float
        east longitude, rad
    model : str, optional
        IGRF implementation: 'ppigrf' (default) or 'igrf'

    Returns
    -------
    b : np.ndarray
        magnetic vector, 1x3 vector, nT, @NED coordinates [north, east, down]

    Notes
    -----
    Requires ppigrf or igrf library:
      pip install ppigrf
      or
      pip install igrf

    References
    ----------
    IGRF-12 (International Geomagnetic Reference Field, 12th generation)

    Revisions
    ---------
    20190216  y.yoshimura

    See also
    --------
    geodetic_igrf
    """
    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)

    if model == 'ppigrf':
        try:
            import ppigrf
            bn, be, bd = ppigrf.igrf(lon_deg, lat_deg, alt, year)
            b = np.array([bn, be, bd]).flatten()
        except ImportError:
            raise ImportError("ppigrf library required. Install with: pip install ppigrf")
    elif model == 'igrf':
        try:
            import igrf
            # igrf library uses different API
            result = igrf.igrf(year, lat_deg, lon_deg, alt)
            b = np.array([result.north, result.east, result.down])
        except ImportError:
            raise ImportError("igrf library required. Install with: pip install igrf")
    else:
        raise ValueError(f"Unknown model: {model}. Use 'ppigrf' or 'igrf'")

    return b


def geodetic_igrf(jd: float, lat: float, lon: float, alt: float,
                  coefs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    # Calculate geocentric magnetic vector with IGRF model

    Parameters
    ----------
    jd : float
        Julian day, day
    lat : float
        geodetic latitude, rad
    lon : float
        geodetic longitude, rad
    alt : float
        geodetic altitude, km
    coefs : np.ndarray, optional
        IGRF coefficients (not used with ppigrf)

    Returns
    -------
    b : np.ndarray
        magnetic field, [northward, eastward, downward], nT

    Notes
    -----
    Wrapper for IGRF model using Julian date input

    References
    ----------
    IGRF-12 (International Geomagnetic Reference Field)

    Revisions
    ---------
    20210428  y.yoshimura
    20260707  y.yoshimura, exact decimal year (fraction of the actual year length)

    See also
    --------
    igrf12
    """
    from ..time_utils import jd2gc
    from ..conversion import gc2jd

    year = jd2gc(jd)[0]
    # decimal year for IGRF coefficient interpolation
    jd_year_start = gc2jd(year, 1, 1, 0, 0, 0)
    year_decimal = year + (jd - jd_year_start) / (gc2jd(year + 1, 1, 1, 0, 0, 0) - jd_year_start)

    b = igrf12(year_decimal, alt, lat, lon)

    return b


# %[appendix]{"version":"1.0"}
