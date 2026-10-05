"""
Jacchia-Roberts 1971 atmospheric model
Python conversion from yMATLAB/environment/jr1971.m
"""

import math
import warnings

import numpy as np

from ..math_utils import wrap_pi


# == Constants =================================================================

_RSTAR = 8.31432       # Universal gas constant [J/(K*mol)]
_AV = 6.022045e23      # Avogadro's constant [molecules/mol]
_RA = 6356.766         # Mean Earth radius [km]
_G0 = 9.80665          # Gravity at sea level [m/s^2]

_M0 = 28.960           # Mean molecular mass at sea level [g/mol]

_T1 = 183.0            # Temperature at lower bound [K]
_Z1 = 90.0             # Altitude of lower bound [km]
_M1 = 28.82678         # Mean molecular mass at z1 [g/mol]
_RHO1 = 3.46e-9        # Density at z1 [g/cm^3]

_Z2 = 100.0            # Second division altitude [km]
_ZX = 125.0            # Inflection point altitude [km]

# Molecular mass [g/mol]: N2, O2, O, Ar, He, H
_MI = (28.0134, 31.9988, 15.9994, 39.9480, 4.0026, 1.00797)

# Thermal diffusion coefficient: N2, O2, O, Ar, He, H
_ALPHAI = (0.0, 0.0, 0.0, 0.0, -0.38, 0.0)

# Fractional volume composition: N2, O2, O, Ar, He, H
_MUI = (0.78110, 0.161778, 0.095544, 0.0093432, 0.61471e-5, 0.0)

_AA = (-435093.363387,
       28275.5646391,
       -765.33466108,
       11.043387545,
       -0.08958790995,
       0.00038737586,
       -0.000000697444)

_CA = (-89284375.0,
       3542400.0,
       -52687.5,
       340.5,
       -0.8)

_LA = (0.1031445e+5,
       0.2341230e+1,
       0.1579202e-2,
       -0.1252487e-5,
       0.2462708e-9)

_ALPHA = (3144902516.672729,
          -123774885.4832917,
          1816141.096520398,
          -11403.31079489267,
          24.36498612105595,
          0.008957502869707995)

_BETA = (-52864482.17910969,
         -16632.50847336828,
         -1.308252378125,
         0.0, 0.0, 0.0)

_ZETA = (0.1985549e-10,
         -0.1833490e-14,
         0.1711735e-17,
         -0.1021474e-20,
         0.3727894e-24,
         -0.7734110e-28,
         0.7026942e-32)

_DELTAIJ = {
    'N2': (0.1093155e2,
           0.1186783e-2,
           -0.1677341e-5,
           0.1420228e-8,
           -0.7139785e-12,
           0.1969715e-15,
           -0.2296182e-19),
    'O2': (0.9924237e1,
           0.1600311e-2,
           -0.2274761e-5,
           0.1938454e-8,
           -0.9782183e-12,
           0.2698450e-15,
           -0.3131808e-19),
    'O': (0.1097083e2,
          0.6118742e-4,
          -0.1165003e-6,
          0.9239354e-10,
          -0.3490739e-13,
          0.5116298e-17,
          0.0),
    'Ar': (0.8049405e1,
           0.2382822e-2,
           -0.3391366e-5,
           0.2909714e-8,
           -0.1481702e-11,
           0.4127600e-15,
           -0.4837461e-19),
    'He': (0.7646886e1,
           -0.4383486e-3,
           0.4694319e-6,
           -0.2894886e-9,
           0.9451989e-13,
           -0.1270838e-16,
           0.0),
}


def jr1971(jd: float, phi_gd: float, lam: float, h: float,
           F10: float, F10a: float, Kp: float) -> dict:
    """
    # Jacchia-Roberts 1971 atmospheric model

    Compute the atmospheric density using the Jacchia-Roberts 1971 model.

    Parameters
    ----------
    jd : float
        Julian day
    phi_gd : float
        geodetic latitude, rad
    lam : float
        longitude, rad
    h : float
        altitude, m
    F10 : float
        10.7-cm solar flux, sfu
    F10a : float
        10.7-cm averaged solar flux, 81-day centered, sfu
    Kp : float
        Kp geomagnetic index (3-hour delayed)

    Returns
    -------
    result : dict
        - total_density: total density, kg/m^3
        - temperature: temperature at the altitude, K
        - exospheric_temperature: exospheric temperature, K
        - N2_number_density, O2_number_density, O_number_density,
          Ar_number_density, He_number_density, H_number_density:
          number densities, 1/m^3

    Notes
    -----
    F10, F10a and Kp can be taken from the CelesTrak SW-All.csv
    (https://celestrak.org/SpaceData/SW-All.csv) with lookup_solar_geo_index:

        F10, F10a, Kp = lookup_solar_geo_index(jd, 'SW-All.csv')

    Example (density at the ISS altitude, ~400 km):

        jd = gc2jd(2024, 1, 1, 12, 0, 0)
        F10, F10a, Kp = lookup_solar_geo_index(jd, 'SW-All.csv')
        result = jr1971(jd, np.deg2rad(35.0), np.deg2rad(139.0), 400e3, F10, F10a, Kp)
        print(result['total_density'])

    References
    ----------
    [1] Roberts, C. R (1971). An analytic model for upper atmosphere
        densities based upon Jacchia's 1970 models.
    [2] Jacchia, L. G (1970). New static models of the thermosphere and
        exosphere with empirical temperature profiles. SAO Special Report #313.
    [3] Vallado, D. A (2013). Fundamentals of Astrodynamics and Applications.
        4th ed. Microcosm Press.
    [4] Long, A. C. et al. (1989). GTDS Mathematical Theory (Revision 1).

    Revisions
    ---------
    20260410  y.yoshimura
    20260929  y.yoshimura, the 3.24 term of the nighttime minimum exospheric
              temperature Tc uses the 81-day average F10a
    20261001  y.yoshimura, wrap_pi instead of the Mapping Toolbox wrapToPi
    20261005  y.yoshimura, fix: factor f in the exponent between 90 and 100 km

    See also
    --------
    lookup_solar_geo_index, load_space_weather, jaccia_bowman
    """
    jd = float(jd)
    phi_gd = float(phi_gd)
    lam = float(lam)
    F10 = float(F10)
    F10a = float(F10a)
    Kp = float(Kp)

    Rstar = _RSTAR
    Av = _AV
    Ra = _RA
    g0 = _G0
    M0 = _M0
    z1 = _Z1
    z2 = _Z2
    T1 = _T1
    M1 = _M1
    rho1 = _RHO1
    zx = _ZX
    Mi = _MI
    alphai = _ALPHAI
    mui = _MUI
    Aa = _AA
    Ca = _CA
    la = _LA
    alpha = _ALPHA
    beta = _BETA
    zeta = _ZETA
    deltaij = _DELTAIJ

    # == Auxiliary variables ===================================================
    Ra2 = Ra * Ra

    # == Preliminaries =========================================================

    # Convert altitude from [m] to [km].
    h = float(h) / 1000

    # Compute the Sun position in MOD frame.
    s_i = _sun_position_mod(jd)

    # Sun declination [rad].
    delta_s = math.atan2(s_i[2], math.sqrt(s_i[0] ** 2 + s_i[1] ** 2))

    # Sun right ascension [rad].
    Omega_s = math.atan2(s_i[1], s_i[0])

    # Right ascension of the selected location w.r.t. inertial frame.
    Omega_p = lam + _jd_to_gmst(jd)

    # Hour angle.
    H = Omega_p - Omega_s

    # == Exospheric Temperature ================================================

    # -- Diurnal Variation -----------------------------------------------------

    dF10 = F10 - F10a
    # 3.24 multiplies the 81-day average (Jacchia 1970 Eq. 14; fixed in
    # SatelliteToolbox 702b621d)
    Tc = 379 + 3.24 * F10a + 1.3 * dF10

    eta = abs(phi_gd - delta_s) / 2
    theta = abs(phi_gd + delta_s) / 2

    tau = float(wrap_pi(H + math.radians(-37 + 6 * math.sin(H + math.radians(43)))))

    Cv = math.cos(eta) ** 2.2
    S = math.sin(theta) ** 2.2
    Tl = Tc * (1 + 0.3 * (S + (Cv - S) * math.cos(tau / 2) ** 3))

    # == Geomagnetic Activity ==================================================

    if h < 200:
        dTinf = 14 * Kp + 0.02 * math.exp(Kp)
    else:
        dTinf = 28 * Kp + 0.03 * math.exp(Kp)

    Tinf = Tl + dTinf

    # == Temperature at the Desired Altitude ===================================

    a_coeff = 371.6678
    b_coeff = 0.0518806
    c_coeff = -294.3505
    d_coeff = -0.00216222
    Tx = a_coeff + b_coeff * Tinf + c_coeff * math.exp(d_coeff * Tinf)

    Tz = _jr1971_temperature(h, Tx, Tinf)

    # == Corrections to the Density ============================================

    # -- Geomagnetic Effect ----------------------------------------------------
    if h < 200:
        dlog10rho_g = 0.012 * Kp + 1.2e-5 * math.exp(Kp)
    else:
        dlog10rho_g = 0.0

    # -- Semi-annual Variation -------------------------------------------------
    Phi = (jd - 2436204.5) / 365.2422

    tau_sa = Phi + 0.09544 * ((0.5 * (1 + math.sin(2 * math.pi * Phi + 6.035))) ** 1.65 - 0.5)
    f_z = (5.876e-7 * h ** 2.331 + 0.06328) * math.exp(-0.002868 * h)
    g_t = (0.02835 + (0.3817 + 0.17829 * math.sin(2 * math.pi * tau_sa + 4.137))
           * math.sin(4 * math.pi * tau_sa + 4.259))

    dlog10rho_sa = f_z * g_t

    # -- Seasonal Latitudinal Variation ----------------------------------------
    sin_phi_gd = math.sin(phi_gd)
    abs_sin_phi_gd = abs(sin_phi_gd)

    dlog10rho_lt = (0.014 * (h - 90) * math.exp(-0.0013 * (h - 90) ** 2)
                    * math.sin(2 * math.pi * Phi + 1.72) * sin_phi_gd * abs_sin_phi_gd)

    # -- Total Correction ------------------------------------------------------
    dlog10rho_c = dlog10rho_g + dlog10rho_lt + dlog10rho_sa
    drho_c = 10 ** dlog10rho_c

    # == Density ===============================================================

    if h == z1:
        rho = rho1 * drho_c

        return _make_output(
            1000 * rho, Tz, Tinf,
            (rho * mui[0]) * Av / Mi[0] * 1e6,
            (rho * mui[1]) * Av / Mi[1] * 1e6,
            (rho * mui[2]) * Av / Mi[2] * 1e6,
            (rho * mui[3]) * Av / Mi[3] * 1e6,
            (rho * mui[4]) * Av / Mi[4] * 1e6,
            (rho * mui[5]) * Av / Mi[5] * 1e6)

    elif z1 < h <= zx:

        # Roots of the polynomial P(Z).
        c0 = (35 ** 4 * Tx / (Tx - T1) + Ca[0]) / Ca[4]
        c1 = Ca[1] / Ca[4]
        c2 = Ca[2] / Ca[4]
        c3 = Ca[3] / Ca[4]
        c4 = 1.0  # Ca[4] / Ca[4]

        r1, r2, xr, yr = _jr1971_roots([c0, c1, c2, c3, c4])

        # -- f and k -----------------------------------------------------------
        f = 35 ** 4 * Ra2 / Ca[4]
        k = -g0 / (Rstar * (Tx - T1))

        # -- X -----------------------------------------------------------------
        X = -2 * r1 * r2 * Ra * (Ra2 + 2 * xr * Ra + xr ** 2 + yr ** 2)

        if h <= z2:

            # == Altitudes Between 90 km and 100 km ============================

            B = [alpha[i] + beta[i] * Tx / (Tx - T1) for i in range(6)]
            B0, B1, B2, B3, B4, B5 = B

            p2 = _evalpoly_asc(r1, B) / _U_func(r1, Ra, xr, yr, r1, r2)
            p3 = -_evalpoly_asc(r2, B) / _U_func(r2, Ra, xr, yr, r1, r2)
            p5 = _evalpoly_asc(-Ra, B) / _V_func(-Ra, xr, yr, r1, r2)

            p4 = (B0 - r1 * r2 * Ra2 * (B4 + (2 * xr + r1 + r2 - Ra) * B5)
                  - r1 * r2 * Ra * (xr ** 2 + yr ** 2) * B5
                  + r1 * r2 * (Ra2 - (xr ** 2 + yr ** 2)) * p5
                  + _W_func(r1, Ra, xr, yr, r1, r2) * p2
                  + _W_func(r2, Ra, xr, yr, r1, r2) * p3) / X

            p6 = (B4 + (2 * xr + r1 + r2 - Ra) * B5 - p5
                  - 2 * (xr + Ra) * p4 - (r2 + Ra) * p3 - (r1 + Ra) * p2)
            p1 = B5 - 2 * p4 - p3 - p2

            # -- F1 and F2 -----------------------------------------------------
            log_F1 = (p1 * math.log((h + Ra) / (z1 + Ra))
                      + p2 * math.log((h - r1) / (z1 - r1))
                      + p3 * math.log((h - r2) / (z1 - r2))
                      + p4 * math.log((h ** 2 - 2 * xr * h + xr ** 2 + yr ** 2)
                                      / (z1 ** 2 - 2 * xr * z1 + xr ** 2 + yr ** 2)))

            F2 = ((h - z1) * (Aa[6] + p5 / ((h + Ra) * (z1 + Ra)))
                  + p6 / yr * math.atan(yr * (h - z1)
                                        / (yr ** 2 + (h - xr) * (z1 - xr))))

            # -- Density -------------------------------------------------------
            Mz = _jr1971_mean_molecular_mass(h)
            # The exponent needs the same factor f as between 100 and 125 km (expk);
            # alpha and beta do not include it. Without f the density is almost
            # constant between 90 and 100 km and jumps by a factor of about 6 at 100 km.
            rho = rho1 * drho_c * Mz * T1 / (M1 * Tz) * math.exp(k * f * (log_F1 + F2))

            return _make_output(
                1000 * rho, Tz, Tinf,
                (rho * mui[0]) * Av / Mi[0] * 1e6,
                (rho * mui[1]) * Av / Mi[1] * 1e6,
                (rho * mui[2]) * Av / Mi[2] * 1e6,
                (rho * mui[3]) * Av / Mi[3] * 1e6,
                (rho * mui[4]) * Av / Mi[4] * 1e6,
                (rho * mui[5]) * Av / Mi[5] * 1e6)

        else:

            # == Altitudes Between 100 km and 125 km ===========================

            T100 = _jr1971_temperature(z2, Tx, Tinf)

            rho100 = _evalpoly_asc(Tinf, zeta) * M0
            rho100 = rho100 * drho_c

            q2 = 1 / _U_func(r1, Ra, xr, yr, r1, r2)
            q3 = -1 / _U_func(r2, Ra, xr, yr, r1, r2)
            q5 = 1 / _V_func(-Ra, xr, yr, r1, r2)
            q4 = (1 + r1 * r2 * (Ra2 - (xr ** 2 + yr ** 2)) * q5
                  + _W_func(r1, Ra, xr, yr, r1, r2) * q2
                  + _W_func(r2, Ra, xr, yr, r1, r2) * q3) / X
            q6 = -q5 - 2 * (xr + Ra) * q4 - (r2 + Ra) * q3 - (r1 + Ra) * q2
            q1 = -2 * q4 - q3 - q2

            log_F3 = (q1 * math.log((h + Ra) / (z2 + Ra))
                      + q2 * math.log((h - r1) / (z2 - r1))
                      + q3 * math.log((h - r2) / (z2 - r2))
                      + q4 * math.log((h ** 2 - 2 * xr * h + xr ** 2 + yr ** 2)
                                      / (z2 ** 2 - 2 * xr * z2 + xr ** 2 + yr ** 2)))

            F4 = (q5 * (h - z2) / ((h + Ra) * (Ra + z2))
                  + q6 / yr * math.atan(yr * (h - z2)
                                        / (yr ** 2 + (h - xr) * (z2 - xr))))

            expk = k * f * (log_F3 + F4)
            rhoN2, rhoO2, rhoO, rhoAr, rhoHe = [
                rho100 * Mi[i] / M0 * mui[i] * (T100 / Tz) ** (1 + alphai[i]) * math.exp(Mi[i] * expk)
                for i in range(5)]

            return _make_output(
                1000 * (rhoN2 + rhoO2 + rhoO + rhoAr + rhoHe), Tz, Tinf,
                rhoN2 * Av / Mi[0] * 1e6,
                rhoO2 * Av / Mi[1] * 1e6,
                rhoO * Av / Mi[2] * 1e6,
                rhoAr * Av / Mi[3] * 1e6,
                rhoHe * Av / Mi[4] * 1e6,
                0.0)

    else:

        # == Altitudes Higher than 125 km ======================================

        rho125 = [drho_c * Mi[i] * 10 ** _evalpoly_asc(Tinf, deltaij[key]) / Av
                  for i, key in enumerate(('N2', 'O2', 'O', 'Ar', 'He'))]

        # -- Compute l ---------------------------------------------------------
        l = _evalpoly_asc(Tinf, la)

        # -- Eq. 25' -----------------------------------------------------------
        gamma = (g0 * Ra2 / (Rstar * l * Tinf) * (Tinf - Tx) / (Tx - T1)
                 * (zx - z1) / (Ra + zx))

        # -- Eq. 25 ------------------------------------------------------------
        rhoN2, rhoO2, rhoO, rhoAr, rhoHe = [
            rho125[i] * (Tx / Tz) ** (1 + alphai[i] + gamma * Mi[i])
            * ((Tinf - Tz) / (Tinf - Tx)) ** (gamma * Mi[i])
            for i in range(5)]

        # -- Helium Seasonal/Latitudinal Correction ----------------------------
        # sign(delta_s) = delta_s / |delta_s| (finite also for delta_s = 0)
        dlog10rho_He = (0.65 / math.radians(23.439291) * abs(delta_s)
                        * (math.sin(math.pi / 4 - phi_gd * np.sign(delta_s) / 2) ** 3 - 0.35355))
        rhoHe = rhoHe * 10 ** dlog10rho_He

        # -- H for Altitude > 500 km -------------------------------------------
        rhoH = 0.0
        if h > 500:
            T500 = _jr1971_temperature(500.0, Tx, Tinf)
            log10_T500 = math.log10(T500)
            rho500_H = Mi[5] / Av * 10 ** (73.13 - (39.4 - 5.5 * log10_T500) * log10_T500)

            gammaH = Mi[5] * gamma
            rhoH = (drho_c * rho500_H * (T500 / Tz) ** (1 + gammaH)
                    * ((Tinf - Tz) / (Tinf - T500)) ** gammaH)

        return _make_output(
            1000 * (rhoN2 + rhoO2 + rhoO + rhoAr + rhoHe + rhoH), Tz, Tinf,
            rhoN2 * Av / Mi[0] * 1e6,
            rhoO2 * Av / Mi[1] * 1e6,
            rhoO * Av / Mi[2] * 1e6,
            rhoAr * Av / Mi[3] * 1e6,
            rhoHe * Av / Mi[4] * 1e6,
            rhoH * Av / Mi[5] * 1e6)


# %[appendix]{"version":"1.0"}


# == Local functions ===========================================================

def _make_output(total_density, temperature, exospheric_temperature,
                 N2, O2, O, Ar, He, H) -> dict:
    return {
        'total_density': float(total_density),
        'temperature': float(temperature),
        'exospheric_temperature': float(exospheric_temperature),
        'N2_number_density': float(N2),
        'O2_number_density': float(O2),
        'O_number_density': float(O),
        'Ar_number_density': float(Ar),
        'He_number_density': float(He),
        'H_number_density': float(H),
    }


def _jr1971_temperature(z: float, Tx: float, Tinf: float) -> float:
    """Compute the temperature [K] at height z [km]."""
    if z < _Z1:
        raise ValueError(f'The altitude must not be lower than {_Z1:.0f} km.')
    if Tinf < 0:
        raise ValueError('The exospheric temperature must be positive.')

    if z <= _ZX:
        aux = _evalpoly_asc(z, _CA)
        T = Tx + (Tx - _T1) / 35 ** 4 * aux
    else:
        l = _evalpoly_asc(Tinf, _LA)
        T = Tinf - (Tinf - Tx) * math.exp(
            -l * ((Tx - _T1) / (Tinf - Tx)) * ((z - _ZX) / (_ZX - _Z1)) / (_RA + z))

    return T


def _jr1971_mean_molecular_mass(z: float) -> float:
    """Compute mean molecular mass at altitude z [km] (valid for 90 <= z <= 100 km)."""
    if z < 90 or z > 100:
        warnings.warn('Empirical mean molecular mass model valid only for 90 <= z <= 100 km.')

    return _evalpoly_asc(z, _AA)


def _jr1971_roots(coeffs) -> tuple:
    """
    Compute roots of the 4th-degree polynomial for density below 125 km.
    coeffs = [c0, c1, c2, c3, c4] (ascending order).
    """
    r = np.roots(coeffs[::-1])  # np.roots expects descending order

    # Separate real and complex roots.
    tol = 1e-10
    real_mask = np.abs(r.imag) < tol
    real_roots = r[real_mask].real
    complex_roots = r[~real_mask]

    r1 = float(np.max(real_roots))
    r2 = float(np.min(real_roots))

    # The complex root with positive imaginary part.
    root = complex_roots[complex_roots.imag > 0][0]
    xr = float(root.real)
    yr = float(abs(root.imag))

    return r1, r2, xr, yr


def _U_func(nu, Ra, x, y, r1, r2):
    return (nu + Ra) ** 2 * (nu ** 2 - 2 * x * nu + x ** 2 + y ** 2) * (r1 - r2)


def _V_func(nu, x, y, r1, r2):
    return (nu ** 2 - 2 * x * nu + x ** 2 + y ** 2) * (nu - r1) * (nu - r2)


def _W_func(nu, Ra, x, y, r1, r2):
    return r1 * r2 * Ra * (Ra + nu) * (Ra + (x ** 2 + y ** 2) / nu)


def _evalpoly_asc(x: float, coeffs) -> float:
    """
    Evaluate polynomial with coefficients in ascending order (c0, c1, c2, ...).
    p(x) = c0 + c1*x + c2*x^2 + ... (Horner's method)
    """
    val = coeffs[-1]
    for c in coeffs[-2::-1]:
        val = val * x + c

    return val


def _sun_position_mod(jd: float) -> tuple:
    """
    Compute the Sun position vector in the Mean-of-Date (MOD) frame [km].
    Low-precision solar coordinates (Vallado, Meeus).
    """
    T_UT1 = (jd - 2451545.0) / 36525.0

    # Mean longitude of the Sun [deg].
    lambda_M = (280.46 + 36000.771 * T_UT1) % 360

    # Mean anomaly of the Sun [deg].
    M_sun = (357.5291092 + 35999.0502909 * T_UT1) % 360
    M_rad = math.radians(M_sun)

    # Ecliptic longitude [deg].
    lambda_ec = lambda_M + 1.914666471 * math.sin(M_rad) + 0.019994643 * math.sin(2 * M_rad)
    lambda_ec_rad = math.radians(lambda_ec)

    # Obliquity of the ecliptic [deg].
    epsilon = 23.439291 - 0.0130042 * T_UT1
    epsilon_rad = math.radians(epsilon)

    # Distance to the Sun [AU].
    r_sun = 1.000140612 - 0.016708617 * math.cos(M_rad) - 0.000139589 * math.cos(2 * M_rad)

    # Convert to km (1 AU = 149597870.7 km).
    r_km = r_sun * 149597870.7

    # Sun position in MOD frame [km].
    return (r_km * math.cos(lambda_ec_rad),
            r_km * math.cos(epsilon_rad) * math.sin(lambda_ec_rad),
            r_km * math.sin(epsilon_rad) * math.sin(lambda_ec_rad))


def _jd_to_gmst(jd: float) -> float:
    """Compute Greenwich Mean Sidereal Time [rad] from Julian Date."""
    T_UT1 = (jd - 2451545.0) / 36525.0

    # GMST in seconds.
    gmst_sec = (67310.54841
                + (876600 * 3600 + 8640184.812866) * T_UT1
                + 0.093104 * T_UT1 ** 2
                - 6.2e-6 * T_UT1 ** 3)

    # Convert to radians (mod 2*pi).
    return (gmst_sec * math.pi / 43200) % (2 * math.pi)
