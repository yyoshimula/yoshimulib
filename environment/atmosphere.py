"""
Atmospheric density models
Python conversion from yMATLAB/environment/
"""

import numpy as np
from typing import Tuple, Optional, Any


class JBData:
    """Container for Jaccia-Bowman 2008 model data"""
    def __init__(self):
        self.SOLdata: Optional[np.ndarray] = None  # Solar activity data
        self.DTCdata: Optional[np.ndarray] = None  # DTC data
        self.EOPdata: Optional[np.ndarray] = None  # Earth Orientation Parameters


def jaccia_bowman(jd: float, lon: float, lat: float, h: float,
                  const: Any, jb_data: JBData) -> Tuple[float, float]:
    """
    # (wrapper) calculating atmospheric density using Jaccia-Bowman model 2008

    Parameters
    ----------
    jd : float
        julian day, day
    lon : float
        geocentric longitude, rad
    lat : float
        geocentric latitude, rad
    h : float
        geodetic height, km
    const : OrbitalConstants
        orbital constants
    jb_data : JBData
        JB2008 coefficients and data

    Returns
    -------
    temp : float
        temperature, K
    rho : float
        air density, kg/m^3

    Notes
    -----
    This is a wrapper function for the JB2008 atmospheric model.
    For full implementation, external JB2008 data files are required.

    For simple applications, consider using nrlmsise00 library:
      pip install nrlmsise00

    References
    ----------
    Bowman, B. R., et al. (2008). A New Empirical Thermospheric Density Model JB2008.
    AIAA/AAS Astrodynamics Specialist Conference.

    Revisions
    ---------
    20221009  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    nrlmsise00_density
    """
    # Note: Full JB2008 implementation requires significant external data
    # and multiple supporting functions (IERS, JPL_Eph_DE430, etc.)
    # This is a placeholder that demonstrates the interface

    raise NotImplementedError(
        "Full JB2008 model requires external data files and supporting functions. "
        "Consider using nrlmsise00_density() for simpler atmospheric density calculations."
    )


def nrlmsise00_density(jd: float, lat: float, lon: float, alt: float,
                       f107: float = 150.0, f107a: float = 150.0,
                       ap: float = 4.0) -> Tuple[float, float]:
    """
    # Calculate atmospheric density using NRLMSISE-00 model

    Parameters
    ----------
    jd : float
        Julian day
    lat : float
        geodetic latitude, rad
    lon : float
        geodetic longitude, rad
    alt : float
        altitude, km
    f107 : float, optional
        daily F10.7 flux (default: 150.0, moderate solar activity)
    f107a : float, optional
        81-day average F10.7 flux (default: 150.0)
    ap : float, optional
        daily magnetic Ap index (default: 4.0, quiet conditions)

    Returns
    -------
    temp : float
        temperature at altitude, K
    rho : float
        total mass density, kg/m^3

    Notes
    -----
    Requires nrlmsise00 library:
      pip install nrlmsise00

    References
    ----------
    Picone, J. M., et al. (2002). NRLMSISE-00 empirical model of the
    atmosphere. Journal of Geophysical Research.

    Revisions
    ---------
    20221009  y.yoshimura

    See also
    --------
    jaccia_bowman
    """
    try:
        from nrlmsise00 import msise_model
    except ImportError:
        raise ImportError("nrlmsise00 library required. Install with: pip install nrlmsise00")

    from ..time_utils import jd2gc

    # Convert JD to date components
    year, month, day, hour, minute, second = jd2gc(jd)
    doy = _day_of_year(year, month, day)
    ut_sec = hour * 3600 + minute * 60 + second

    # Convert to degrees
    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)

    # Call NRLMSISE-00
    # Returns: [He, O, N2, O2, Ar, Total mass density, H, N, Anomalous O]
    # and temperatures [exospheric temp, temperature at altitude]
    densities, temperatures = msise_model(
        jd,            # datetime or year
        alt,           # altitude, km
        lat_deg,       # latitude, deg
        lon_deg,       # longitude, deg
        f107a,         # 81-day average F10.7
        f107,          # daily F10.7
        ap,            # daily magnetic index
        lst=ut_sec/3600 + lon_deg/15  # local apparent solar time
    )

    rho = densities[5] * 1e3  # Convert g/cm^3 to kg/m^3
    temp = temperatures[1]    # Temperature at altitude

    return temp, rho


def _day_of_year(year: int, month: int, day: int) -> int:
    """Calculate day of year from date components"""
    import datetime
    d = datetime.date(year, month, day)
    return d.timetuple().tm_yday


def exponential_atmosphere(h: float, h0: float = 0.0, rho0: float = 1.225,
                           H: float = 8.5) -> float:
    """
    # Simple exponential atmospheric density model

    Parameters
    ----------
    h : float
        altitude, km
    h0 : float, optional
        reference altitude, km (default: 0)
    rho0 : float, optional
        density at reference altitude, kg/m^3 (default: 1.225, sea level)
    H : float, optional
        scale height, km (default: 8.5)

    Returns
    -------
    rho : float
        atmospheric density, kg/m^3

    Notes
    -----
    Simple exponential model: rho = rho0 * exp(-(h-h0)/H)
    Only valid for rough estimates, especially at low altitudes.

    Revisions
    ---------
    20221009  y.yoshimura
    """
    rho = rho0 * np.exp(-(h - h0) / H)
    return rho


# Standard atmosphere scale heights for different altitude ranges (km)
SCALE_HEIGHTS = {
    (0, 25): 7.249,
    (25, 30): 6.349,
    (30, 40): 6.682,
    (40, 50): 7.554,
    (50, 60): 8.382,
    (60, 70): 7.714,
    (70, 80): 6.549,
    (80, 90): 5.799,
    (90, 100): 5.382,
    (100, 110): 5.877,
    (110, 120): 7.263,
    (120, 130): 9.473,
    (130, 140): 12.636,
    (140, 150): 16.149,
    (150, 180): 22.523,
    (180, 200): 29.740,
    (200, 250): 37.105,
    (250, 300): 45.546,
    (300, 350): 53.628,
    (350, 400): 53.298,
    (400, 450): 58.515,
    (450, 500): 60.828,
    (500, 600): 63.822,
    (600, 700): 71.835,
    (700, 800): 88.667,
    (800, 900): 124.64,
    (900, 1000): 181.05,
}


# %[appendix]{"version":"1.0"}
