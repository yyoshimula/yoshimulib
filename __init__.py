"""
yoshimulib - Aerospace Engineering Library for Spacecraft Dynamics

Python conversion from yMATLAB/
A comprehensive library for spacecraft dynamics, attitude control, orbital mechanics,
and space environment modeling.

Modules:
--------
attitude
    Attitude dynamics, DCM, quaternions, Euler angles, kinematics
conversion
    Unit and calendar conversions
math_utils
    Mathematical utilities (skew matrix, wrap_pi, Legendre polynomials)
time_utils
    Time system conversions (JD, MJD, UTC, TT, leap seconds)
dual_quaternions
    Dual quaternion operations for rigid body motion
geometric_integration
    Lie group integrators preserving SO(3) structure
gpr
    Gaussian Process Regression
spherical_gaussian
    Spherical Gaussian functions for BRDF modeling
ukf_ckf
    Unscented/Cubature Kalman Filter implementations
object
    3D spacecraft model I/O (OBJ/MTL files)
orbit
    Orbital mechanics, Kepler's equation, coordinate transforms
orbit_determination
    Initial orbit determination (Gibbs, Gauss, Double-R)
relative_orbit
    Relative orbital elements, HCW equations
srp
    Solar Radiation Pressure models
lightcurves
    Synthetic light curve generation using BRDF models
environment
    Space environment models (IGRF-12, Jaccia-Bowman, Jacchia-Roberts 1971)
sun_moon
    Solar and lunar ephemerides
hifi_srp
    High-fidelity SRP models
spice
    SPICE integration (use SpiceyPy for full functionality)
utility
    Visualization and plotting utilities

Author: Yasuhiro Yoshimura (y.yoshimula@gmail.com)
"""

__version__ = "1.0.0"
__author__ = "Yasuhiro Yoshimura"

def passfail(passed: bool) -> str:
    """
    # MATLAB-compatible helper (MATLAB: passfail)
    """
    return "PASS" if passed else "FAIL"

# Import key modules for convenient access
from . import attitude
from . import conversion
from . import math_utils
from . import time_utils
from . import dual_quaternions
from . import geometric_integration
from . import gpr
from . import spherical_gaussian
from . import ukf_ckf
from . import object
from . import orbit
from . import orbit_determination
from . import relative_orbit
from . import srp
from . import lightcurves
from . import environment
from . import sun_moon
from . import hifi_srp
from . import spice
from . import utility

__all__ = [
    'attitude',
    'conversion',
    'math_utils',
    'time_utils',
    'dual_quaternions',
    'geometric_integration',
    'gpr',
    'spherical_gaussian',
    'ukf_ckf',
    'object',
    'orbit',
    'orbit_determination',
    'relative_orbit',
    'srp',
    'lightcurves',
    'environment',
    'sun_moon',
    'hifi_srp',
    'spice',
    'utility',
    'passfail',
]
