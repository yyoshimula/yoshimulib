"""
Sun and Moon ephemeris calculations
Python conversion from yMATLAB/sunMoon/
"""

import numpy as np
import warnings
from pathlib import Path
from ..conversion import au2km


def earth_vsop87(jd: np.ndarray, earth_vsop: np.ndarray) -> tuple:
    """
    # Earth heliocentric longitude, latitude, and distance

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, n x 1 vector
    earth_vsop : np.ndarray
        VSOP87 coefficients for Earth

    Returns
    -------
    lon : np.ndarray
        Earth's heliocentric longitude, rad, n x 1 vector
    lat : np.ndarray
        Earth's heliocentric latitude, rad, n x 1
    r : np.ndarray
        Earth's heliocentric distance, AU, n x 1

    Notes
    -----
    Calculates Earth's heliocentric longitude, latitude, and distance for a
    given Julian day number, referred to the mean ecliptic and equinox of date.

    References
    ----------
    Jean Meeus, "Astronomical Algorithms, 2nd edition", p. 217.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    vsop_const
    """
    jd = np.atleast_1d(jd).flatten()
    t = (jd - 2451545.0) / 365250.0  # 1 x m vector

    # Index ranges (0-indexed)
    l_ind = slice(0, 130)
    b_ind = slice(130, 137)
    r_ind = slice(137, 196)

    ser = earth_vsop[:, 1]  # n x 1 vector
    # Broadcasting: earth_vsop[:, 4:5] is n x 1, t is 1 x m
    tmp = earth_vsop[:, 3:4] * np.cos(earth_vsop[:, 4:5] + earth_vsop[:, 5:6] * t)

    L_tmp = tmp[l_ind, :]
    L0 = np.sum(L_tmp * (ser[l_ind, np.newaxis] == 0), axis=0)
    L1 = np.sum(L_tmp * (ser[l_ind, np.newaxis] == 1), axis=0)
    L2 = np.sum(L_tmp * (ser[l_ind, np.newaxis] == 2), axis=0)
    L3 = np.sum(L_tmp * (ser[l_ind, np.newaxis] == 3), axis=0)
    L4 = np.sum(L_tmp * (ser[l_ind, np.newaxis] == 4), axis=0)
    L5 = np.sum(L_tmp * (ser[l_ind, np.newaxis] == 5), axis=0)

    B_tmp = tmp[b_ind, :]
    B0 = np.sum(B_tmp * (ser[b_ind, np.newaxis] == 0), axis=0)
    B1 = np.sum(B_tmp * (ser[b_ind, np.newaxis] == 1), axis=0)

    R_tmp = tmp[r_ind, :]
    R0 = np.sum(R_tmp * (ser[r_ind, np.newaxis] == 0), axis=0)
    R1 = np.sum(R_tmp * (ser[r_ind, np.newaxis] == 1), axis=0)
    R2 = np.sum(R_tmp * (ser[r_ind, np.newaxis] == 2), axis=0)
    R3 = np.sum(R_tmp * (ser[r_ind, np.newaxis] == 3), axis=0)
    R4 = np.sum(R_tmp * (ser[r_ind, np.newaxis] == 4), axis=0)

    lon_val = L0 + t * L1 + t ** 2 * L2 + t ** 3 * L3 + t ** 4 * L4 + t ** 5 * L5
    lon = lon_val / 1e8

    lat_val = B0 + t * B1
    lat = lat_val / 1e8

    r_val = R0 + t * R1 + t ** 2 * R2 + t ** 3 * R3 + t ** 4 * R4
    r = r_val / 1e8

    return lon, lat, r


def sun_lon_lat_r(jd: np.ndarray, const, earth_vsop: np.ndarray) -> tuple:
    """
    # Sun's geocentric longitude, latitude, and distance

    Referred to the mean ecliptic and equinox of date.

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, n x 1 vector
    const : OrbitalConstants
        orbital constants
    earth_vsop : np.ndarray
        Earth VSOP87 coefficients

    Returns
    -------
    lon : np.ndarray
        Sun's geocentric longitude, rad
    lat : np.ndarray
        Sun's geocentric latitude, rad
    r : np.ndarray
        Sun's geocentric distance, AU

    Notes
    -----
    To obtain the geocentric longitude and latitude of the Sun, add 180 (deg) to
    Earth's heliocentric longitude and change the sign of Earth's heliocentric latitude.

    References
    ----------
    Jean Meeus, "Astronomical Algorithms, 2nd edition", p.166 and p. 217.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    orbit_const, vsop_const, precession
    """
    from ..orbit.precession import precession

    # Earth's heliocentric longitude, latitude, and distance
    lon, lat, r = earth_vsop87(jd, earth_vsop)

    lon = lon + np.pi
    lat = -1 * lat

    _, _, _, eta, Pi_, p = precession(const.J2000, jd, const)

    A = np.sin(eta) * np.sin(lat) + np.cos(eta) * np.cos(lat) * np.sin(p + Pi_ - lon)
    B = np.cos(lat) * np.cos(p + Pi_ - lon)
    C = np.cos(eta) * np.sin(lat) - np.sin(eta) * np.cos(lat) * np.sin(p + Pi_ - lon)

    lon = Pi_ - np.arctan2(A, B)
    lat = np.arcsin(C)

    return lon, lat, r


def sun(jd: np.ndarray, const, earth_vsop: np.ndarray) -> np.ndarray:
    """
    # Sun position w.r.t. J2000.0 frame

    Parameters
    ----------
    jd : np.ndarray
        Julian date of UTC, day, n x 1 vector
    const : OrbitalConstants
        orbital constants
    earth_vsop : np.ndarray
        Earth VSOP87 coefficients

    Returns
    -------
    sun_pos : np.ndarray
        Sun position w.r.t. J2000.0 frame, km, n x 3 matrix

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sun_lon_lat_r, earth_vsop87, vsop_const
    """
    from ..attitude import q_rotation

    jd = np.atleast_1d(jd).flatten()

    # Sun geocentric longitude, latitude, and distance w.r.t.
    # mean ecliptic and equinox of J2000.0 frame
    lon_s, lat_s, sun_au = sun_lon_lat_r(jd, const, earth_vsop)
    tmp = au2km(sun_au, const) * np.column_stack([
        np.cos(lat_s) * np.cos(lon_s),
        np.cos(lat_s) * np.sin(lon_s),
        np.sin(lat_s)
    ])

    # Conversion: r_i_sun = R_3(-eps0) r_sun
    # Sun position vector at inertial frame (J2000.0 frame), km
    tmp_q = np.array([np.sin(-const.EPS0 / 2), 0, 0, np.cos(-const.EPS0 / 2)])
    tmp_q = np.tile(tmp_q, (tmp.shape[0], 1))
    sun_pos = q_rotation(tmp, tmp_q, scalar=4)  # n x 3

    return sun_pos


def moon_elp(jd: np.ndarray, elp: dict) -> tuple:
    """
    # Moon's geocentric longitude, latitude, and distance

    Referred to the mean ecliptic and equinox of date.

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, scalar or array
    elp : dict
        ELP coefficients with keys 'a' and 'b'

    Returns
    -------
    lon : np.ndarray
        Moon's geocentric longitude, rad
    lat : np.ndarray
        Moon's geocentric latitude, rad
    r : np.ndarray
        Moon's geocentric distance, km

    References
    ----------
    Jean Meeus, "Astronomical Algorithms, 2nd edition", p. 337.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com
    """
    jd = np.atleast_1d(jd).flatten()

    # Set coefficients for longitude and distance
    coeff_d = elp['a'][:, 0]
    coeff_m = elp['a'][:, 1]
    coeff_mp = elp['a'][:, 2]
    coeff_f = elp['a'][:, 3]
    sum_l = elp['a'][:, 4]
    sum_r = elp['a'][:, 5]

    # For latitude
    coeff_d_lat = elp['b'][:, 0]
    coeff_m_lat = elp['b'][:, 1]
    coeff_mp_lat = elp['b'][:, 2]
    coeff_f_lat = elp['b'][:, 3]
    sum_b = elp['b'][:, 4]

    # Calculate
    t = (jd - 2451545.0) / 36525.0

    # Mean longitude, deg
    l_prime = 218.3164477 + 481267.88123421 * t - 0.0015786 * t ** 2 + t ** 3 / 538841.0 - t ** 4 / 65194000.0
    l_prime = np.mod(l_prime, 360)

    # Mean elongation, deg
    D = 297.8501921 + t * 445267.1114034 - t ** 2 * 0.0018819 + t ** 3 / 545868.0 - t ** 4 / 113065000.0
    D = np.mod(D, 360)

    # Sun's mean anomaly, deg
    M = 357.5291092 + t * 35999.0502909 - t ** 2 * 0.0001536 + t ** 3 / 24490000.0
    M = np.mod(M, 360)

    # Moon's mean anomaly, deg
    M_prime = 134.9633964 + t * 477198.8675055 + t ** 2 * 0.0087414 + t ** 3 / 69699.0 - t ** 4 / 14712000.0
    M_prime = np.mod(M_prime, 360)

    # Moon's argument of latitude, deg
    F = 93.2720950 + t * 483202.0175233 - t ** 2 * 0.0036539 - t ** 3 / 3526000.0 + t ** 4 / 863310000.0
    F = np.mod(F, 360)

    l_prime = np.deg2rad(l_prime)
    D = np.deg2rad(D)
    M = np.deg2rad(M)
    M_prime = np.deg2rad(M_prime)
    F = np.deg2rad(F)

    # Additional terms
    A = np.array([
        np.mod(119.75 + 131.849 * t, 360.0),
        np.mod(53.09 + 479264.290 * t, 360.0),
        np.mod(313.45 + 481266.484 * t, 360.0)
    ])
    A = np.deg2rad(A)

    # Eq. (47.6)
    E = 1.0 - t * 0.002516 - t ** 2 * 0.0000074

    # Longitude and distance
    # Broadcasting: coeff_d is (n,), D is (m,) -> need outer product
    arg = (coeff_d[:, np.newaxis] * D + coeff_m[:, np.newaxis] * M +
           coeff_mp[:, np.newaxis] * M_prime + coeff_f[:, np.newaxis] * F)
    tmp_lon = np.sin(arg)
    tmp_r = np.cos(arg)

    # Apply E factors
    abs_coeff_m = np.abs(coeff_m)[:, np.newaxis]
    E_factor = np.where(abs_coeff_m == 0, 1,
                        np.where(abs_coeff_m == 1, E, E ** 2))

    lon = np.sum(sum_l[:, np.newaxis] * E_factor * tmp_lon, axis=0)
    r = np.sum(sum_r[:, np.newaxis] * E_factor * tmp_r, axis=0)

    # Latitude
    arg_lat = (coeff_d_lat[:, np.newaxis] * D + coeff_m_lat[:, np.newaxis] * M +
               coeff_mp_lat[:, np.newaxis] * M_prime + coeff_f_lat[:, np.newaxis] * F)
    tmp_lat = np.sin(arg_lat)

    abs_coeff_m_lat = np.abs(coeff_m_lat)[:, np.newaxis]
    E_factor_lat = np.where(abs_coeff_m_lat == 0, 1,
                            np.where(abs_coeff_m_lat == 1, E, E ** 2))

    lat = np.sum(sum_b[:, np.newaxis] * E_factor_lat * tmp_lat, axis=0)

    # Additive terms, deg
    lon = lon + 3958 * np.sin(A[0]) + 1962 * np.sin(l_prime - F) + 318 * np.sin(A[1])

    lat = (lat - 2235 * np.sin(l_prime) + 382 * np.sin(A[2]) + 175 * np.sin(A[0] - F)
           + 175 * np.sin(A[0] + F) + 127 * np.sin(l_prime - M_prime) - 115 * np.sin(l_prime + M_prime))

    lon = l_prime + np.deg2rad(lon / 1e6)  # rad
    lat = np.deg2rad(lat / 1e6)  # rad
    r = 385000.56 + (r / 1e3)  # km

    return lon, lat, r


def moon_lon_lat_r(jd: np.ndarray, const, elp: dict) -> tuple:
    """
    # Moon's geocentric longitude, latitude, and distance

    Referred to the mean ecliptic and equinox of J2000.0.

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, scalar or array
    const : OrbitalConstants
        orbital constants
    elp : dict
        ELP coefficients

    Returns
    -------
    lon_m : np.ndarray
        Moon's geocentric longitude, rad
    lat_m : np.ndarray
        Moon's geocentric latitude, rad
    r_m : np.ndarray
        Moon's geocentric distance, km

    References
    ----------
    M. Chapront-Touzé and J. Chapront. The lunar ephemeris, ELP 2000.
    Astronomy and Astrophysics, vol. 124, 1983, pp. 50-62.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sun
    """
    from ..orbit.precession import precession

    # Moon position w.r.t. ELP 2000 frame
    lon, lat, r_m = moon_elp(jd, elp)

    _, _, _, eta, Pi_, p = precession(const.J2000, jd, const)

    # Conversion
    A = np.sin(eta) * np.sin(lat) + np.cos(eta) * np.cos(lat) * np.sin(p + Pi_ - lon)
    B = np.cos(lat) * np.cos(p + Pi_ - lon)
    C = np.cos(eta) * np.sin(lat) - np.sin(eta) * np.cos(lat) * np.sin(p + Pi_ - lon)

    lon_m = Pi_ - np.arctan2(A, B)
    lat_m = np.arcsin(C)

    return lon_m, lat_m, r_m


def moon(jd: np.ndarray, const, elp: dict) -> np.ndarray:
    """
    # Moon position w.r.t. J2000.0 frame

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, scalar or array
    const : OrbitalConstants
        orbital constants
    elp : dict
        ELP coefficients

    Returns
    -------
    moon_pos : np.ndarray
        Moon position w.r.t. J2000.0 frame, km, n x 3 matrix

    References
    ----------
    M. Chapront-Touzé and J. Chapront. The lunar ephemeris, ELP 2000.
    Astronomy and Astrophysics, vol. 124, 1983, pp. 50-62.

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    sun
    """
    from ..attitude import q_rotation

    jd = np.atleast_1d(jd).flatten()

    # Moon geocentric longitude, latitude, and distance w.r.t.
    # mean ecliptic and equinox of J2000.0 frame
    lon_m, lat_m, r_m = moon_lon_lat_r(jd, const, elp)
    tmp = r_m[:, np.newaxis] * np.column_stack([
        np.cos(lat_m) * np.cos(lon_m),
        np.cos(lat_m) * np.sin(lon_m),
        np.sin(lat_m)
    ])

    # Conversion: r_i_moon = R_3(-eps0) r_moon
    # Moon position vector at inertial frame (J2000.0 frame), km
    tmp_q = np.array([np.sin(-const.EPS0 / 2), 0, 0, np.cos(-const.EPS0 / 2)])
    tmp_q = np.tile(tmp_q, (tmp.shape[0], 1))
    moon_pos = q_rotation(tmp, tmp_q, scalar=4)  # n x 3

    return moon_pos


def sun_g(jd: np.ndarray, r_vec: np.ndarray, const, earth_vsop: np.ndarray,
          use_spice: bool = False) -> tuple:
    """
    # Sun's gravitational force

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, n x 1 vector
    r_vec : np.ndarray
        satellite position at inertial frame, km, n x 3 vector
    const : OrbitalConstants
        orbital constants
    earth_vsop : np.ndarray
        Earth VSOP87 coefficients
    use_spice : bool
        use SPICE (default: False)

    Returns
    -------
    a_sun : np.ndarray
        Sun's gravitational force, km/s^2, n x 3 vector
    sun_i : np.ndarray
        Sun's position at inertial frame (GCRF), km, n x 3 vector

    Notes
    -----
    ELP is not necessary if SPICE is used

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    orbit_const, vsop_const, precession
    """
    r_vec = np.atleast_2d(r_vec)

    if not use_spice:
        sun_i = sun(jd, const, earth_vsop)
    else:
        raise NotImplementedError("SPICE support requires spiceypy library")

    # Relative position vector from satellite to sun at inertial frame
    r_sc2s = sun_i - r_vec  # n x 3

    # At inertial frame
    r_sc2s_norm = np.linalg.norm(r_sc2s, axis=1, keepdims=True)
    sun_i_norm = np.linalg.norm(sun_i, axis=1, keepdims=True)

    a_sun = const.GS * (r_sc2s / r_sc2s_norm ** 3 - sun_i / sun_i_norm ** 3)

    return a_sun, sun_i


def moon_g(jd: np.ndarray, r_vec: np.ndarray, const, elp: dict,
           use_spice: bool = False) -> tuple:
    """
    # Moon gravitational force

    Parameters
    ----------
    jd : np.ndarray
        Julian day, day, n x 1 vector
    r_vec : np.ndarray
        satellite position at inertial frame, km, n x 3 vector
    const : OrbitalConstants
        orbital constants
    elp : dict
        ELP coefficients (not necessary if SPICE is used)
    use_spice : bool
        use SPICE (default: False)

    Returns
    -------
    a_moon : np.ndarray
        Moon gravitational force, km/s^2, n x 3 vector
    moon_ijk : np.ndarray
        Moon position at inertial frame, km, n x 3 vector

    Notes
    -----
    ELP is not necessary if SPICE is used

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    orbit_const, sun_g
    """
    r_vec = np.atleast_2d(r_vec)

    if not use_spice:
        moon_ijk = moon(jd, const, elp)
    else:
        raise NotImplementedError("SPICE support requires spiceypy library")

    # Relative position vector from satellite to moon at inertial frame
    r_sc2m = moon_ijk - r_vec  # n x 3

    # Acceleration at inertial frame
    r_sc2m_norm = np.linalg.norm(r_sc2m, axis=1, keepdims=True)
    moon_norm = np.linalg.norm(moon_ijk, axis=1, keepdims=True)

    a_moon = const.GM * (r_sc2m / r_sc2m_norm ** 3 - moon_ijk / moon_norm ** 3)

    return a_moon, moon_ijk


# ELP2000 coefficients (simplified)
# For full precision, load from file
ELP_COEFF_A = np.array([
    [0, 0, 1, 0, 6288774, -20905355],
    [2, 0, -1, 0, 1274027, -3699111],
    [2, 0, 0, 0, 658314, -2955968],
    [0, 0, 2, 0, 213618, -569925],
    [0, 1, 0, 0, -185116, 48888],
    [0, 0, 0, 2, -114332, -3149],
    [2, 0, -2, 0, 58793, 246158],
    [2, -1, -1, 0, 57066, -152138],
    [2, 0, 1, 0, 53322, -170733],
    [2, -1, 0, 0, 45758, -204586],
    [0, 1, -1, 0, -40923, -129620],
    [1, 0, 0, 0, -34720, 108743],
    [0, 1, 1, 0, -30383, 104755],
    [2, 0, 0, -2, 15327, 10321],
    [0, 0, 1, 2, -12528, 0],
    [0, 0, 1, -2, 10980, 79661],
    [4, 0, -1, 0, 10675, -34782],
    [0, 0, 3, 0, 10034, -23210],
    [4, 0, -2, 0, 8548, -21636],
    [2, 1, -1, 0, -7888, 24208],
])

ELP_COEFF_B = np.array([
    [0, 0, 0, 1, 5128122],
    [0, 0, 1, 1, 280602],
    [0, 0, 1, -1, 277693],
    [2, 0, 0, -1, 173237],
    [2, 0, -1, 1, 55413],
    [2, 0, -1, -1, 46271],
    [2, 0, 0, 1, 32573],
    [0, 0, 2, 1, 17198],
    [2, 0, 1, -1, 9266],
    [0, 0, 2, -1, 8822],
    [2, -1, 0, -1, 8216],
    [2, 0, -2, -1, 4324],
    [2, 0, 1, 1, 4200],
    [2, 1, 0, -1, -3359],
    [2, -1, -1, 1, 2463],
    [2, -1, 0, 1, 2211],
    [2, -1, -1, -1, 2065],
    [0, 1, -1, -1, -1870],
    [4, 0, -1, -1, 1828],
    [0, 1, 0, 1, -1794],
])

# Default ELP coefficients
ELP_DEFAULT = {
    'a': ELP_COEFF_A,
    'b': ELP_COEFF_B
}


def read_elp(fnames: list[str] | None = None,
             data_dir: str | None = None,
             fallback_default: bool = True) -> dict:
    """
    # Read ELP coefficients for Moon (MATLAB: readELP)

    Parameters
    ----------
    fnames : list[str], optional
        list of two filenames [A, B]
    data_dir : str, optional
        directory containing the files
    fallback_default : bool
        use built-in coefficients if files are not found
    """
    if fnames is None:
        fnames = ['ELPcoeffA.csv', 'ELPcoeffB.csv']
    if len(fnames) != 2:
        raise ValueError("fnames must contain two filenames: [A, B]")

    if data_dir is not None:
        candidates = [Path(data_dir)]
    else:
        candidates = [
            Path(__file__).resolve().parent,
            Path(__file__).resolve().parent.parent / 'orbit',
            Path(__file__).resolve().parent.parent / 'data',
            Path(__file__).resolve().parent.parent / 'yMATLAB' / 'orbit',
        ]

    path_a = None
    path_b = None
    for base in candidates:
        a = base / fnames[0]
        b = base / fnames[1]
        if a.exists() and b.exists():
            path_a = a
            path_b = b
            break

    if path_a is None or path_b is None:
        if fallback_default:
            warnings.warn(
                "ELP coefficient files not found; using built-in defaults.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                'a': ELP_COEFF_A.copy(),
                'b': ELP_COEFF_B.copy(),
            }
        raise FileNotFoundError(
            f"ELP coefficient files not found in {candidates}"
        )

    elp_a = np.loadtxt(path_a, delimiter=',')
    elp_b = np.loadtxt(path_b, delimiter=',')
    return {'a': elp_a, 'b': elp_b}


def readELP(fnames: list[str] | None = None) -> dict:
    """
    # MATLAB-compatible alias of read_elp
    """
    return read_elp(fnames=fnames)


# %[appendix]{"version":"1.0"}
