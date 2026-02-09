"""
orbit - Orbital mechanics and coordinate transformations

yoshimuLibrary orbit module
Python conversion from yMATLAB/orbit/
"""

from .kepler import (
    kepler_eq,
    e_anomaly,
    true_anomaly,
    mean_anomaly,
)

from .orbital_elements import (
    oe2rv,
    rv2oe,
    MU_EARTH_KM,
    MU_EARTH_M,
    MU_SUN_KM,
    MU_MOON_KM,
    R_EARTH_KM,
    R_EARTH_M,
    J2_EARTH,
    AU_KM,
)

from .constants import (
    OrbitalConstants,
    orbit_const,
    VSOP87_EARTH,
    earth_vsop87,
    vsop_const,
    vsopConst,
)

from .precession import (
    precession,
    precession_dcm,
    precession_q,
    nutation,
    nutation_dcm,
    nutation_q,
    earth_nutation_precession_q,
    earthNutationPrecessionQ,
    obliquity,
)

from .sidereal import (
    gmst,
    gast,
    era,
)

from .transforms import (
    earth_w,
    wobble,
    shadow,
    dcm_i2rtn,
    ecef2lat_lon_h,
    geocentric2geodetic,
    geodetic2geocentric,
    earth_g,
)

from .gravity import (
    read_egm2008,
    egm2008,
)

from .jb2008 import (
    read_jb2008,
    readJB2008,
)

from .mee import (
    OrbitalElements,
    coe2mee,
    mee2coe,
    calc_orbital_state,
    mee_derivatives,
    mee,
    gve,
    mean2osc,
)

from .frame_transforms import (
    EOPData,
    IAU06Data,
    read_iau06,
    read_eop,
    get_eop,
    eop,
    jd2jdt,
    pef2itrf,
    precession_nutation,
    itrf2gcrf,
    q_itrf2gcrf,
    tod2mod,
    mod2j2000,
    teme2j2000,
    earth_full_rot_q,
)

from .verify_mee import (
    initializeConfig,
    runPropagationAnalysis,
    calcDCM,
    eomGVE,
    eomMEE,
    extractMEE,
    extractState,
)

from .tle import (
    TLEData,
    read_tle,
    read_tle_single,
)

__all__ = [
    # kepler
    'kepler_eq', 'e_anomaly', 'true_anomaly', 'mean_anomaly',
    # orbital elements
    'oe2rv', 'rv2oe',
    # constants
    'MU_EARTH_KM', 'MU_EARTH_M', 'MU_SUN_KM', 'MU_MOON_KM',
    'R_EARTH_KM', 'R_EARTH_M', 'J2_EARTH', 'AU_KM',
    'OrbitalConstants', 'orbit_const', 'VSOP87_EARTH', 'earth_vsop87',
    'vsop_const', 'vsopConst',
    # precession
    'precession', 'precession_dcm', 'precession_q',
    'nutation', 'nutation_dcm', 'nutation_q',
    'earth_nutation_precession_q', 'earthNutationPrecessionQ', 'obliquity',
    # sidereal
    'gmst', 'gast', 'era',
    # transforms
    'earth_w', 'wobble', 'shadow', 'dcm_i2rtn',
    'ecef2lat_lon_h', 'geocentric2geodetic', 'geodetic2geocentric', 'earth_g',
    # gravity
    'read_egm2008', 'egm2008',
    # JB2008
    'read_jb2008', 'readJB2008',
    # MEE and GVE
    'OrbitalElements', 'coe2mee', 'mee2coe', 'calc_orbital_state',
    'mee_derivatives', 'mee', 'gve', 'mean2osc',
    # frame transforms
    'EOPData', 'IAU06Data', 'read_iau06', 'read_eop', 'get_eop', 'eop',
    'jd2jdt', 'pef2itrf', 'precession_nutation',
    'itrf2gcrf', 'q_itrf2gcrf',
    'tod2mod', 'mod2j2000', 'teme2j2000', 'earth_full_rot_q',
    # TLE
    'TLEData', 'read_tle', 'read_tle_single',
    # verifyMEE helpers
    'initializeConfig', 'runPropagationAnalysis', 'calcDCM', 'eomGVE', 'eomMEE',
    'extractMEE', 'extractState',
]
