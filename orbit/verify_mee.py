"""
verifyMEE helpers
Python conversion from yMATLAB/orbit/verifyMEE.m (local functions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..attitude import dcm1axis, q2dcm
from ..conversion import gc2jd, s2day
from ..orbit import VSOP87_EARTH
from ..orbit.gravity import egm2008
from ..orbit.mee import OrbitalElements, coe2mee, mee2coe, calc_orbital_state, gve, mee
from ..orbit.transforms import dcm_i2rtn
from ..sun_moon import sun_g, moon_g, ELP_DEFAULT
from ..srp.srp_models import srp_cannon
from ..orbit.kepler import true_anomaly
from ..orbit.frame_transforms import itrf2gcrf


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_egm(const) -> Dict[str, Any]:
    egm = _get(const, 'EGM', None)
    if egm is None:
        return {}
    if isinstance(egm, dict):
        return egm
    return {
        'GEODEG': getattr(egm, 'GEODEG', None),
        'Cnm': getattr(egm, 'Cnm', None),
        'Snm': getattr(egm, 'Snm', None),
    }


def initializeConfig() -> Dict[str, Any]:
    """
    # Configuration defaults (MATLAB: initializeConfig)
    """
    config: Dict[str, Any] = {}
    config['geodeg'] = 8
    config['anomalyFlag'] = 1
    config['jdIni'] = gc2jd(2021, 1, 1, 12, 0, 0)
    config['dt'] = 60 * 5  # seconds
    config['tspanOrbits'] = 10
    config['opts'] = {'rtol': 1e-8, 'atol': 1e-8}
    config['cannonAm'] = 20 / 3000
    config['figWindowNum'] = 1
    return config


def extractState(x_: np.ndarray, anomaly_flag: int) -> OrbitalElements:
    """
    # Extract orbital elements from state vector (MATLAB: extractState)
    """
    x_ = np.asarray(x_).flatten()
    oe = OrbitalElements()
    oe.a = float(x_[0])
    oe.e = float(x_[1])
    oe.inc = float(x_[2])
    oe.raan = float(x_[3])
    oe.w = float(x_[4])

    if anomaly_flag == 1:
        oe.nu = float(x_[5])
    else:
        oe.nu = float(true_anomaly(oe.a, oe.e, x_[5])[0])

    oe.raan = np.mod(oe.raan, 2 * np.pi)
    oe.w = np.mod(oe.w, 2 * np.pi)
    oe.nu = np.mod(oe.nu, 2 * np.pi)
    return oe


def extractMEE(x_: np.ndarray) -> OrbitalElements:
    """
    # Extract modified equinoctial elements (MATLAB: extractMEE)
    """
    x_ = np.asarray(x_).flatten()
    oe = OrbitalElements()
    oe.p_ = float(x_[0])
    oe.f_ = float(x_[1])
    oe.g_ = float(x_[2])
    oe.h_ = float(x_[3])
    oe.k_ = float(x_[4])
    oe.L_ = float(x_[5])
    return oe


def calcDCM(oe: OrbitalElements, atti: Any, para: Dict[str, Any], const, t_: float):
    """
    # Calculate DCMs (MATLAB: calcDCM)
    """
    recef_i = dcm1axis(3, const.WE * t_) @ para['Rini']
    ri_ecef = recef_i.T

    roi = dcm_i2rtn(oe.raan, oe.inc, oe.w, oe.nu)

    if atti is None:
        rbi = np.eye(3)
    else:
        q = _get(atti, 'q', None)
        rbi = q2dcm(q, scalar=4) if q is not None else np.eye(3)

    return roi, rbi, ri_ecef


def eomGVE(t_: float, x_: np.ndarray, para: Dict[str, Any],
           chief: Any, const, config: Dict[str, Any]) -> np.ndarray:
    """
    # Equations of Motion for GVE (MATLAB: eomGVE)
    """
    oe = extractState(x_, para['flag'])
    oe = calc_orbital_state(oe, const.GE)
    r_vec = oe.rVec.reshape(3,)

    roi, _, ri_ecef = calcDCM(oe, None, para, const, t_)

    a_rtn = np.zeros(3)

    egm = _get_egm(const)
    if egm.get('GEODEG') is not None and egm.get('Cnm') is not None and egm.get('Snm') is not None:
        tmp = egm2008(r_vec, int(egm['GEODEG']), egm['Cnm'], egm['Snm'], const)
        a_earth = ri_ecef @ tmp.reshape(3, 1)
        a_earth = roi @ a_earth
        a_rtn += a_earth.flatten()

    jd = para['jdIni'] + s2day(t_)
    earth_vsop = _get(const, 'earthVSOP', VSOP87_EARTH)
    a_sun_i, sun_i = sun_g(jd, r_vec.reshape(1, 3), const, earth_vsop)
    a_rtn += (roi @ a_sun_i.reshape(3, 1)).flatten()

    elp = _get(const, 'ELP', ELP_DEFAULT)
    a_moon, _ = moon_g(jd, r_vec.reshape(1, 3), const, elp)
    a_rtn += (roi @ a_moon.reshape(3, 1)).flatten()

    sun_rel_i = sun_i.reshape(3,) - r_vec
    sun_dist = np.linalg.norm(sun_rel_i)
    a_srp = srp_cannon(config['cannonAm'], sun_rel_i.reshape(1, 3), sun_dist * 1e3, const)
    a_srp = a_srp.reshape(-1) / 1e3
    a_rtn += a_srp

    d_orbital_state = gve(oe, a_rtn, para['flag'], const.GE)
    return np.asarray(d_orbital_state).flatten()


def eomMEE(t_: float, x_: np.ndarray, para: Dict[str, Any],
           const, config: Dict[str, Any]) -> np.ndarray:
    """
    # Equations of Motion for MEE (MATLAB: eomMEE)
    """
    oe = extractMEE(x_)
    oe = mee2coe(oe)
    oe = calc_orbital_state(oe, const.GE)
    r_vec = oe.rVec.reshape(3,)

    roi, _, ri_ecef = calcDCM(oe, None, para, const, t_)

    a_rtn = np.zeros(3)

    egm = _get_egm(const)
    if egm.get('GEODEG') is not None and egm.get('Cnm') is not None and egm.get('Snm') is not None:
        tmp = egm2008(r_vec, int(egm['GEODEG']), egm['Cnm'], egm['Snm'], const)
        a_earth = ri_ecef @ tmp.reshape(3, 1)
        a_earth = roi @ a_earth
        a_rtn += a_earth.flatten()

    jd = para['jdIni'] + s2day(t_)
    earth_vsop = _get(const, 'earthVSOP', VSOP87_EARTH)
    a_sun_i, sun_i = sun_g(jd, r_vec.reshape(1, 3), const, earth_vsop)
    a_rtn += (roi @ a_sun_i.reshape(3, 1)).flatten()

    elp = _get(const, 'ELP', ELP_DEFAULT)
    a_moon, _ = moon_g(jd, r_vec.reshape(1, 3), const, elp)
    a_rtn += (roi @ a_moon.reshape(3, 1)).flatten()

    sun_rel_i = sun_i.reshape(3,) - r_vec
    sun_dist = np.linalg.norm(sun_rel_i)
    a_srp = srp_cannon(config['cannonAm'], sun_rel_i.reshape(1, 3), sun_dist * 1e3, const)
    a_srp = a_srp.reshape(-1) / 1e3
    a_rtn += a_srp

    d_orbital_state = mee(oe, a_rtn, const.GE)
    return np.asarray(d_orbital_state).flatten()


def runPropagationAnalysis(config: Dict[str, Any], chief: Any, const) -> Dict[str, Any]:
    """
    # Run propagation analysis (MATLAB: runPropagationAnalysis)
    """
    from scipy.integrate import solve_ivp

    tspan = np.arange(0, _get(chief, 'T') * config['tspanOrbits'] + config['dt'], config['dt'])

    para = {
        'Rini': itrf2gcrf(config['jdIni'], _get(const, 'EOP')),
        'jdIni': config['jdIni'],
        'flag': config['anomalyFlag'],
    }

    chief_oe = _get(chief, 'oe')
    if isinstance(chief_oe, OrbitalElements):
        oe_obj = chief_oe
    elif isinstance(chief_oe, dict):
        oe_obj = OrbitalElements(**chief_oe)
    else:
        oe_obj = chief_oe

    mee_ini_obj = coe2mee(oe_obj)
    mee_ini = np.array([mee_ini_obj.p_, mee_ini_obj.f_, mee_ini_obj.g_,
                        mee_ini_obj.h_, mee_ini_obj.k_, mee_ini_obj.L_])

    sol_mee = solve_ivp(
        lambda t, x: eomMEE(t, x, para, const, config),
        (tspan[0], tspan[-1]),
        mee_ini,
        t_eval=tspan,
        rtol=config['opts'].get('rtol', 1e-8),
        atol=config['opts'].get('atol', 1e-8),
    )

    oe_ini = np.asarray(_get(chief, 'oeIni')).flatten()
    sol_gve = solve_ivp(
        lambda t, x: eomGVE(t, x, para, chief, const, config),
        (tspan[0], tspan[-1]),
        oe_ini,
        t_eval=tspan,
        rtol=config['opts'].get('rtol', 1e-8),
        atol=config['opts'].get('atol', 1e-8),
    )

    results = {
        'meeData': sol_mee.y.T,
        'coeData': sol_gve.y.T,
        'tspan': tspan,
    }
    return results

