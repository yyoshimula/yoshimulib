"""
Reference frame transformations (ITRF, GCRF, TEME, TOD, MOD, J2000)
Python conversion from yMATLAB/orbit/itrf2gcrf.m, teme2J2000.m, etc.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EOPData:
    """Earth Orientation Parameters data container"""
    mjd: float = 0.0
    xp: float = 0.0   # polar motion x, rad
    yp: float = 0.0   # polar motion y, rad
    dUT1: float = 0.0 # UT1-UTC, seconds
    dX: float = 0.0   # celestial pole offset X, rad
    dY: float = 0.0   # celestial pole offset Y, rad
    lod: float = 0.0  # length of day, seconds


@dataclass
class IAU06Data:
    """IAU 2006 precession-nutation data"""
    axs0: np.ndarray = field(default_factory=lambda: np.array([]))
    a0xi: np.ndarray = field(default_factory=lambda: np.array([]))
    ays0: np.ndarray = field(default_factory=lambda: np.array([]))
    a0yi: np.ndarray = field(default_factory=lambda: np.array([]))
    ass0: np.ndarray = field(default_factory=lambda: np.array([]))
    a0si: np.ndarray = field(default_factory=lambda: np.array([]))


def read_iau06(data_dir: Optional[str] = None) -> IAU06Data:
    """
    # Read IAU06 values for precession and nutation

    Parameters
    ----------
    data_dir : str, optional
        directory containing iau06x.dat, iau06y.dat, iau06s.dat

    Returns
    -------
    iau06 : IAU06Data
        IAU 2006 coefficients

    Notes
    -----
    iau06x.dat, iau06y.dat, iau06s.dat are required.

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition, p.213

    Revisions
    ---------
    20230608  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..conversion import arcs2rad

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    else:
        data_dir = Path(data_dir)

    iau06 = IAU06Data()

    # Read X coefficients
    iau06x = np.loadtxt(data_dir / 'iau06x.dat')
    iau06.axs0 = iau06x[:, 1:3]
    iau06.a0xi = iau06x[:, 3:17]
    iau06.axs0 = 1e-6 * arcs2rad(iau06.axs0)  # micro arcsec to rad

    # Read Y coefficients
    iau06y = np.loadtxt(data_dir / 'iau06y.dat')
    iau06.ays0 = iau06y[:, 1:3]
    iau06.a0yi = iau06y[:, 3:17]
    iau06.ays0 = 1e-6 * arcs2rad(iau06.ays0)

    # Read S coefficients
    iau06s = np.loadtxt(data_dir / 'iau06s.dat')
    iau06.ass0 = iau06s[:, 1:3]
    iau06.a0si = iau06s[:, 3:17]
    iau06.ass0 = 1e-6 * arcs2rad(iau06.ass0)

    return iau06


def read_eop(filename: Optional[str] = None, data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    # Read EOP data

    Parameters
    ----------
    filename : str, optional
        EOP file name (default: EOP_20_C04_one_file_1962-now.txt)
    data_dir : str, optional
        directory containing the EOP file

    Returns
    -------
    eop : dict
        EOP data including dataAll, iau06, leapJD

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition, p.213

    Revisions
    ---------
    20230608  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..time_utils import leap_s

    if filename is None:
        filename = 'EOP_20_C04_one_file_1962-now.txt'

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data'
    else:
        data_dir = Path(data_dir)

    file_path = data_dir / filename

    # Read EOP data
    data_all = np.loadtxt(file_path)

    eop = {
        'dataAll': data_all,
        'iau06': read_iau06(data_dir),
        'leapJD': leap_s()
    }

    return eop


def get_eop(year: int, month: int, day: int, eop_data_all: np.ndarray) -> EOPData:
    """
    # Get Earth orientation parameters for a specific date

    Parameters
    ----------
    year : int
        year
    month : int
        month
    day : int
        day
    eop_data_all : np.ndarray
        EOP data array from read_eop

    Returns
    -------
    eop : EOPData
        Earth orientation parameters

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition.

    Revisions
    ---------
    20230605  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..conversion import arcs2rad
    from ..time_utils import jd2mjd
    from ..conversion import gc2jd

    # Julian day to modified Julian day
    jd = gc2jd(year, month, day, 0, 0, 0)
    mjd = jd2mjd(jd)

    # Find index
    idx = np.where(eop_data_all[:, 4] == mjd)[0]

    if len(idx) == 0:
        raise ValueError(f"No EOP data for MJD {mjd}")

    idx = idx[0]

    eop = EOPData()
    eop.mjd = eop_data_all[idx, 4]
    eop.xp = arcs2rad(eop_data_all[idx, 5])
    eop.yp = arcs2rad(eop_data_all[idx, 6])
    eop.dUT1 = eop_data_all[idx, 7]
    eop.dX = arcs2rad(eop_data_all[idx, 8])
    eop.dY = arcs2rad(eop_data_all[idx, 9])
    eop.lod = eop_data_all[idx, 12]

    return eop


def eop(year: int, month: int, day: int, eop_data_all) -> EOPData:
    """
    # MATLAB-compatible wrapper for EOP extraction (MATLAB: eop)

    Parameters
    ----------
    year : int
        year
    month : int
        month
    day : int
        day
    eop_data_all : array-like or dict
        EOP data array or dict containing 'dataAll'/'data'
    """
    if isinstance(eop_data_all, dict):
        if 'dataAll' in eop_data_all:
            data = eop_data_all['dataAll']
        elif 'data' in eop_data_all:
            data = eop_data_all['data']
        else:
            raise KeyError("eop_data_all dict must contain 'dataAll' or 'data'")
    elif hasattr(eop_data_all, 'data'):
        data = eop_data_all.data
    else:
        data = eop_data_all

    return get_eop(year, month, day, np.asarray(data))


def jd2jdt(jd: float) -> float:
    """
    # Julian day to Julian century

    Parameters
    ----------
    jd : float
        Julian day

    Returns
    -------
    t : float
        Julian century from J2000.0
    """
    return (jd - 2451545.0) / 36525.0


def pef2itrf(xp: float, yp: float) -> np.ndarray:
    """
    # DCM from PEF to ITRF

    Parameters
    ----------
    xp : float
        polar motion x, rad
    yp : float
        polar motion y, rad

    Returns
    -------
    R : np.ndarray
        direction cosine matrix, 3x3

    Notes
    -----
    This function is the rotation from PEF to ITRF.
    Vallado's textbook uses ITRF to PEF, but this function is PEF to ITRF.

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition, p.223. Eq. (3-78)

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com
    """
    R = np.array([
        [1, 0, xp],
        [0, 1, -yp],
        [-xp, yp, 1]
    ])
    return R


def precession_nutation(jd_tt: float, iau06: IAU06Data,
                        dX: float = 0.0, dY: float = 0.0) -> np.ndarray:
    """
    # Precession and nutation based on IAU-2006/2000 theory

    Calculate DCM from CIRS to GCRF.

    Parameters
    ----------
    jd_tt : float
        Julian day of Terrestrial Time
    iau06 : IAU06Data
        IAU 2006 coefficients
    dX : float
        correction term X, rad (optional)
    dY : float
        correction term Y, rad (optional)

    Returns
    -------
    dcm : np.ndarray
        DCM from CIRS to GCRF, 3x3

    Notes
    -----
    iau06x.dat, iau06y.dat, iau06s.dat are required.

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition, p.213

    Revisions
    ---------
    20230608  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..conversion import arcs2rad
    from ..attitude import dcm1axis

    t_tt = jd2jdt(jd_tt)  # Julian centuries of TT

    # Delaunay fundamental arguments, deg
    l = (134.96340251 + (1717915923.2178 * t_tt + 31.8792 * t_tt**2
         + 0.051635 * t_tt**3 - 0.00024470 * t_tt**4) / 3600.0)
    l1 = (357.52910918 + (129596581.0481 * t_tt - 0.5532 * t_tt**2
          - 0.000136 * t_tt**3 - 0.00001149 * t_tt**4) / 3600.0)
    f = (93.27209062 + (1739527262.8478 * t_tt - 12.7512 * t_tt**2
         + 0.001037 * t_tt**3 + 0.00000417 * t_tt**4) / 3600.0)
    d = (297.85019547 + (1602961601.2090 * t_tt - 6.3706 * t_tt**2
         + 0.006593 * t_tt**3 - 0.00003169 * t_tt**4) / 3600.0)
    omega = (125.04455501 + (-6962890.5431 * t_tt + 7.4722 * t_tt**2
             + 0.007702 * t_tt**3 - 0.00005939 * t_tt**4) / 3600.0)

    # Planetary arguments, deg
    lonmer = 252.250905494 + 149472.6746358 * t_tt
    lonven = 181.979800853 + 58517.8156748 * t_tt
    lonear = 100.466448494 + 35999.3728521 * t_tt
    lonmar = 355.433274605 + 19140.299314 * t_tt
    lonjup = 34.351483900 + 3034.90567464 * t_tt
    lonsat = 50.0774713998 + 1222.11379404 * t_tt
    lonurn = 314.055005137 + 428.466998313 * t_tt
    lonnep = 304.348665499 + 218.486200208 * t_tt
    precrate = 1.39697137214 * t_tt + 0.0003086 * t_tt**2

    # Convert to radians
    l = np.deg2rad(np.mod(l, 360.0))
    l1 = np.deg2rad(np.mod(l1, 360.0))
    f = np.deg2rad(np.mod(f, 360.0))
    d = np.deg2rad(np.mod(d, 360.0))
    omega = np.deg2rad(np.mod(omega, 360.0))
    lonmer = np.deg2rad(np.mod(lonmer, 360.0))
    lonven = np.deg2rad(np.mod(lonven, 360.0))
    lonear = np.deg2rad(np.mod(lonear, 360.0))
    lonmar = np.deg2rad(np.mod(lonmar, 360.0))
    lonjup = np.deg2rad(np.mod(lonjup, 360.0))
    lonsat = np.deg2rad(np.mod(lonsat, 360.0))
    lonurn = np.deg2rad(np.mod(lonurn, 360.0))
    lonnep = np.deg2rad(np.mod(lonnep, 360.0))
    precrate = np.deg2rad(np.mod(precrate, 360.0))

    args = np.array([l, l1, f, d, omega, lonmer, lonven, lonear, lonmar,
                     lonjup, lonsat, lonurn, lonnep, precrate])

    # Calculate X
    def calc_sum(a0i, axs0, indices, args):
        tmp = np.sum(a0i[indices, :] * args, axis=1)
        return np.sum(axs0[indices, 0] * np.sin(tmp) + axs0[indices, 1] * np.cos(tmp))

    # X sums
    xsum0 = calc_sum(iau06.a0xi, iau06.axs0, np.arange(1306), args)
    xsum1 = calc_sum(iau06.a0xi, iau06.axs0, np.arange(1306, 1306 + 253), args)
    xsum2 = calc_sum(iau06.a0xi, iau06.axs0, np.arange(1306 + 253, 1306 + 253 + 36), args)
    xsum3 = calc_sum(iau06.a0xi, iau06.axs0, np.arange(1306 + 253 + 36, 1306 + 253 + 36 + 4), args)
    xsum4 = calc_sum(iau06.a0xi, iau06.axs0, np.arange(1306 + 253 + 36 + 4, 1306 + 253 + 36 + 5), args)

    X = (-0.016617 + 2004.191898 * t_tt - 0.4297829 * t_tt**2
         - 0.19861834 * t_tt**3 - 0.000007578 * t_tt**4 + 0.0000059285 * t_tt**5)
    X = arcs2rad(X) + xsum0 + xsum1 * t_tt + xsum2 * t_tt**2 + xsum3 * t_tt**3 + xsum4 * t_tt**4

    # Y sums
    ysum0 = calc_sum(iau06.a0yi, iau06.ays0, np.arange(962), args)
    ysum1 = calc_sum(iau06.a0yi, iau06.ays0, np.arange(962, 962 + 277), args)
    ysum2 = calc_sum(iau06.a0yi, iau06.ays0, np.arange(962 + 277, 962 + 277 + 30), args)
    ysum3 = calc_sum(iau06.a0yi, iau06.ays0, np.arange(962 + 277 + 30, 962 + 277 + 30 + 5), args)
    ysum4 = calc_sum(iau06.a0yi, iau06.ays0, np.arange(962 + 277 + 30 + 5, 962 + 277 + 30 + 6), args)

    Y = (-0.006951 - 0.025896 * t_tt - 22.4072747 * t_tt**2
         + 0.00190059 * t_tt**3 + 0.001112526 * t_tt**4 + 0.0000001358 * t_tt**5)
    Y = arcs2rad(Y) + ysum0 + ysum1 * t_tt + ysum2 * t_tt**2 + ysum3 * t_tt**3 + ysum4 * t_tt**4

    # S sums
    ssum0 = calc_sum(iau06.a0si, iau06.ass0, np.arange(33), args)
    ssum1 = calc_sum(iau06.a0si, iau06.ass0, np.arange(33, 33 + 3), args)
    ssum2 = calc_sum(iau06.a0si, iau06.ass0, np.arange(33 + 3, 33 + 3 + 25), args)
    ssum3 = calc_sum(iau06.a0si, iau06.ass0, np.arange(33 + 3 + 25, 33 + 3 + 25 + 4), args)
    ssum4 = calc_sum(iau06.a0si, iau06.ass0, np.arange(33 + 3 + 25 + 4, 33 + 3 + 25 + 5), args)

    s = (0.000094 + 0.00380865 * t_tt - 0.00012268 * t_tt**2
         - 0.07257411 * t_tt**3 + 0.00002798 * t_tt**4 + 0.00001562 * t_tt**5)
    s = -X * Y * 0.5 + arcs2rad(s) + ssum0 + ssum1 * t_tt + ssum2 * t_tt**2 + ssum3 * t_tt**3 + ssum4 * t_tt**4

    # Add corrections
    X = X + dX
    Y = Y + dY

    # Calculate a
    a = 0.5 + 0.125 * (X * X + Y * Y)

    # DCM
    dcm_tmp = np.array([
        [1 - a * X**2, -a * X * Y, X],
        [-a * X * Y, 1 - a * Y**2, Y],
        [-X, -Y, 1 - a * (X**2 + Y**2)]
    ])

    dcm = dcm_tmp @ dcm1axis(3, s)

    return dcm


def itrf2gcrf(jd: float, eop: Dict[str, Any]) -> np.ndarray:
    """
    # Rotation matrix from ITRF to GCRF (CIO approach of IAU-2006/2000 reduction)

    Parameters
    ----------
    jd : float
        Julian day (UTC)
    eop : dict
        EOP data from read_eop

    Returns
    -------
    dcm : np.ndarray
        DCM from ITRF to GCRF, 3x3

    Notes
    -----
    位置座標の変換のみ。速度を変換する際は地球の自転速度を考慮した計算が必要

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition, p.220

    Revisions
    ---------
    20230612  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..time_utils import jd2gc, dat, utc2tt
    from ..conversion import gc2jd
    from ..attitude import dcm1axis
    from .sidereal import era
    from .transforms import wobble

    # Time and EOP
    yy, mm, dd, hh, mi, ss = jd2gc(jd)
    delta_at = dat(jd, eop['leapJD'])
    eop_data = get_eop(int(yy), int(mm), int(dd), eop['dataAll'])

    # UT1 = UTC + dUT1
    # Convert dUT1 from seconds to days and add to jd
    jd_ut1 = jd + eop_data.dUT1 / 86400.0

    jd_tt = utc2tt(jd, delta_at)
    t_tt = jd2jdt(jd_tt)

    # DCM from ITRF to TIRS
    s_prime = wobble(t_tt)
    itrf2tirs = dcm1axis(3, -s_prime) @ dcm1axis(2, eop_data.xp) @ dcm1axis(1, eop_data.yp)

    # Earth rotation angle
    theta_era = era(jd_ut1)

    # DCM from TIRS to CIRS
    tirs2cirs = dcm1axis(3, -theta_era)

    # Precession and nutation (DCM from CIRS to GCRF)
    cirs2gcrf = precession_nutation(jd_tt, eop['iau06'], eop_data.dX, eop_data.dY)

    dcm = cirs2gcrf @ tirs2cirs @ itrf2tirs

    return dcm


def q_itrf2gcrf(scalar: int, jd: np.ndarray, eop: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    # Quaternion from ITRF to GCRF

    Parameters
    ----------
    scalar : int
        quaternion scalar position (0 or 4)
    jd : np.ndarray
        Julian day array
    eop : dict, optional
        EOP data from read_eop

    Returns
    -------
    q : np.ndarray
        quaternion from ITRF to GCRF, n x 4

    Notes
    -----
    itrf2gcrfのwrapper的な関数

    References
    ----------
    Vallado, D. A., & McClain, W. D. (2001). Fundamentals of Astrodynamics
    and Applications. 4th edition, p.220

    Revisions
    ---------
    20230612  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..attitude import dcm2q

    if eop is None:
        eop = read_eop()

    jd = np.atleast_1d(jd)
    q = np.zeros((len(jd), 4))

    for i, jd_i in enumerate(jd):
        dcm = itrf2gcrf(jd_i, eop)
        q[i, :] = dcm2q(scalar, dcm)

    return q


def tod2mod(jd: float, inc: float, raan: float, w: float, const) -> Tuple[float, float, float]:
    """
    # Converting TOD frame to MOD frame

    Parameters
    ----------
    jd : float
        Julian day
    inc : float
        inclination, rad
    raan : float
        longitude of the ascending node, rad
    w : float
        argument of the perihelion, rad
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    i_mod : float
        inclination at MOD, rad
    raan_mod : float
        longitude of the ascending node at MOD, rad
    w_mod : float
        argument of the perihelion at MOD, rad

    References
    ----------
    David A. Vallado, "Fundamentals of Astrodynamics and Applications, 3rd ed.," pp.229-231.
    Jean Meeus, "Astronomical Algorithms, 2nd ed.," pp.143-148.

    Revisions
    ---------
    20210601  y.yoshimura
    """
    from .precession import nutation_dcm

    dcm = nutation_dcm(jd, const)

    tmp = np.array([
        np.sin(raan) * np.sin(inc),
        -np.cos(raan) * np.sin(inc),
        np.cos(inc)
    ])

    i_mod = np.arccos(dcm[:, 2] @ tmp)
    raan_mod = np.arctan2(dcm[:, 0] @ tmp, dcm[:, 1] @ (-tmp))

    w_mod = w - np.arctan2(
        -np.cos(raan) * dcm[0, 2] - np.sin(raan) * dcm[1, 2],
        -np.sin(raan) * np.cos(inc) * dcm[0, 2] + np.cos(raan) * np.cos(inc) * dcm[1, 2] + np.sin(inc) * dcm[2, 2]
    )

    w_mod = np.mod(w_mod, 2 * np.pi)
    raan_mod = np.mod(raan_mod, 2 * np.pi)

    return i_mod, raan_mod, w_mod


def mod2j2000(jd: float, inc: float, raan: float, w: float, const) -> Tuple[float, float, float]:
    """
    # Converting MOD frame to J2000 frame

    Parameters
    ----------
    jd : float
        Julian day
    inc : float
        inclination, rad
    raan : float
        longitude of the ascending node, rad
    w : float
        argument of the perihelion, rad
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    i_j : float
        inclination at J2000.0, rad
    raan_j : float
        longitude of the ascending node at J2000.0, rad
    w_j : float
        argument of the perihelion at J2000.0, rad

    References
    ----------
    David A. Vallado, "Fundamentals of Astrodynamics and Applications, 3rd ed.," pp.229-231.
    Jean Meeus, "Astronomical Algorithms, 2nd ed.," pp.143-148.

    Revisions
    ---------
    20210601  y.yoshimura
    """
    from .precession import precession

    zeta, z, theta, _, _, _ = precession(const.J2000, jd, const)

    i_j = np.arccos(np.cos(theta) * np.cos(inc) - np.sin(theta) * np.sin(raan - z) * np.sin(inc))

    raan_j = np.arctan2(
        np.cos(theta) * np.sin(raan - z) * np.sin(inc) + np.sin(theta) * np.cos(inc),
        np.cos(raan - z) * np.sin(inc)
    ) - zeta

    w_j = w - np.arctan2(
        np.sin(theta) * np.cos(raan - z),
        np.cos(theta) * np.sin(inc) + np.sin(theta) * np.sin(raan - z) * np.cos(inc)
    )

    raan_j = np.mod(raan_j, 2 * np.pi)
    w_j = np.mod(w_j, 2 * np.pi)

    return i_j, raan_j, w_j


def teme2j2000(jd: float, inc: float, raan: float, w: float, const) -> Tuple[float, float, float]:
    """
    # Converting TEME frame to J2000 frame

    Parameters
    ----------
    jd : float
        Julian day
    inc : float
        inclination, rad
    raan : float
        longitude of the ascending node, rad
    w : float
        argument of the perihelion, rad
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    i_j : float
        inclination at J2000.0, rad
    raan_j : float
        longitude of the ascending node at J2000.0, rad
    w_j : float
        argument of the perihelion at J2000.0, rad

    References
    ----------
    David A. Vallado, "Fundamentals of Astrodynamics and Applications, 3rd ed.," pp.229-231.

    Revisions
    ---------
    2021020209  y.yoshimura
    """
    from .precession import obliquity, nutation

    # TEME to TOD
    epsi = obliquity(jd)
    d_psi, d_epsi = nutation(jd, const)
    raan = raan + d_psi * np.cos(epsi + d_epsi)

    # TOD -> MOD -> J2000
    i_mod, raan_mod, w_mod = tod2mod(jd, inc, raan, w, const)
    i_j, raan_j, w_j = mod2j2000(jd, i_mod, raan_mod, w_mod, const)

    return i_j, raan_j, w_j


def earth_full_rot_q(jd0: float, jd1: np.ndarray, scalar: int, const) -> np.ndarray:
    """
    # Earth rotation quaternion including precession and nutation

    Parameters
    ----------
    jd0 : float
        Julian day at epoch
    jd1 : np.ndarray
        Julian day array
    scalar : int
        quaternion scalar position (0 or 4)
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    q : np.ndarray
        quaternion from epoch jd0 to jd1, n x 4

    References
    ----------
    NA

    Revisions
    ---------
    20211027  y.yoshimura, y.yoshimula@gmail.com
    """
    from ..attitude import q_mult
    from .precession import precession_q, nutation_q

    jd1 = np.atleast_1d(jd1)

    # Precession
    q_p = precession_q(jd0, jd1, scalar, const)

    # Nutation
    q_n = nutation_q(jd1, scalar, const)

    # TOD = Nutation * Precession * J2000
    q = q_mult(q_n, q_p, scalar=scalar, definition=1)

    return q


# %[appendix]{"version":"1.0"}
