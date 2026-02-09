"""
Gravity models including EGM2008
Python conversion from yMATLAB/orbit/egm2008.m, readEGM2008.m
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from ..math_utils import associated_legendre


def read_egm2008(deg: int, filename: Optional[str] = None,
                 normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Reading EGM2008 coefficients

    Parameters
    ----------
    deg : int
        degree to be read
    filename : str, optional
        coefficient file name (default: EGM2008_to2190_TideFree.txt)
    normalized : bool
        if True, outputs are normalized coefficients (default: False)

    Returns
    -------
    Cnm : np.ndarray
        coefficients C, (deg+1) x (deg+1) matrix
    Snm : np.ndarray
        coefficients S, (deg+1) x (deg+1) matrix

    Notes
    -----
    引数でnormalized=Trueを指定したときだけnormalized coefficientsを出力
    egm2008関数ではunnormalized coefficientsを使用する

    References
    ----------
    David A. Vallado, "Fundamentals of Astrodynamics and Applications, 4th edition, pp.538-550.
    Montenbruck Oliver & Eberhard Gill, Satellite Orbits. Springer Science & Business Media 2012., pp.61-67

    Revisions
    ---------
    20210502  y.yoshimura

    See also
    --------
    egm2008
    """
    if filename is None:
        filename = 'EGM2008_to2190_TideFree.txt'

    Cnm = np.zeros((deg + 1, deg + 1))
    Snm = np.zeros((deg + 1, deg + 1))

    # Try to find the file
    file_path = Path(filename)
    if not file_path.exists():
        # Try in the data directory
        data_dir = Path(__file__).parent.parent / 'data'
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"EGM2008 coefficient file not found: {filename}\n"
                f"Please provide the full path or place the file in the data directory."
            )

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                n = int(parts[0])
                m = int(parts[1])
            except ValueError:
                continue

            if n > deg:
                break

            # Parse scientific notation (handles D instead of E)
            c_str = parts[2].replace('D', 'E').replace('d', 'e')
            s_str = parts[3].replace('D', 'E').replace('d', 'e')

            try:
                c_val = float(c_str)
                s_val = float(s_str)
            except ValueError:
                # Try parsing format like "1.23D-04" split across columns
                try:
                    c_val = float(parts[2]) * 10 ** float(parts[3])
                    s_val = float(parts[4]) * 10 ** float(parts[5])
                except (ValueError, IndexError):
                    continue

            Cnm[n, m] = c_val
            Snm[n, m] = s_val

            if not normalized:
                # Transform to unnormalized coefficients
                if m == 0:
                    k = 1
                else:
                    k = 2
                from math import factorial
                Pi = np.sqrt(factorial(n + m) / (factorial(n - m) * k * (2 * n + 1)))
                Cnm[n, m] = Cnm[n, m] / Pi
                Snm[n, m] = Snm[n, m] / Pi

    return Cnm, Snm


def egm2008(r_vec: np.ndarray, deg: int, Cnm: np.ndarray, Snm: np.ndarray,
            const) -> np.ndarray:
    """
    # Calculating the non-spherical Earth's gravitational attraction with EGM2008

    Parameters
    ----------
    r_vec : np.ndarray
        object's position at ECEF frame, 1x3 vector, km
    deg : int
        Degree and order required (up to degree and order 20)
    Cnm : np.ndarray
        Earth's gravitational coefficients C of EGM2008
    Snm : np.ndarray
        Earth's gravitational coefficients S of EGM2008
    const : OrbitalConstants
        orbital constants

    Returns
    -------
    a : np.ndarray
        perturbing accelerations in ECEF frame (Cartesian coordinates), 1x3 vector, km/s^2

    Notes
    -----
    EGM2008を用いた高次の地球重力加速度

    References
    ----------
    David A. Vallado, "Fundamentals of Astrodynamics and Applications, 4th edition, pp.538-550.
    Montenbruck Oliver & Eberhard Gill, Satellite Orbits. Springer Science & Business Media 2012., pp.61-67

    Revisions
    ---------
    20230119 arguments changed, y.yoshimura
    20210419  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    read_egm2008, orbit_const
    """
    r_vec = np.asarray(r_vec).flatten()
    x, y, z = r_vec[0], r_vec[1], r_vec[2]
    r = np.linalg.norm(r_vec)

    phi = np.arcsin(z / r)  # geocentric latitude
    lam = np.arctan2(y, x)  # longitude
    sp = np.sin(phi)

    # Associated Legendre polynomials
    P = np.zeros((deg + 1, deg + 1))
    for i in range(1, deg + 1):
        tmp = np.arange(i + 1)
        # yoshimuLibrary's legendre function
        P[i, :i + 1] = ((-1.0) ** tmp) * associated_legendre(i, sp)

    P[0, :] = 0
    P[1, :] = 0  # Only use degree >= 2

    # Accelerations (except for two-body acceleration)
    # Partial derivatives of potential
    tmp_lam = lam * np.arange(deg + 1)
    tmpC = np.tile(np.cos(tmp_lam), (deg + 1, 1))  # cos(m*lam) matrix
    tmpS = np.tile(np.sin(tmp_lam), (deg + 1, 1))  # sin(m*lam) matrix

    n = np.arange(deg + 1).reshape(-1, 1)  # degree of geopotential (column vector)

    # dU/dr
    tmpCoef = (const.RE / r) ** n * (n + 1) * P * (Cnm * tmpC + Snm * tmpS)
    dudr = -const.GE / r ** 2 * np.sum(tmpCoef)

    # dU/dphi
    P2 = np.column_stack([P[:, 1:], np.zeros(deg + 1)])  # P(n,m+1)
    m = np.tile(np.arange(deg + 1), (deg + 1, 1))  # order of geopotential
    tmpCoef = (const.RE / r) ** n * (P2 - np.tan(phi) * m * P) * (Cnm * tmpC + Snm * tmpS)
    dudphi = const.GE / r * np.sum(tmpCoef)

    # dU/dlambda
    tmpCoef = (const.RE / r) ** n * m * P * (-Cnm * tmpS + Snm * tmpC)
    dudlam = const.GE / r * np.sum(tmpCoef)

    # Accelerations w.r.t. Earth-fixed frame (except for two-body acceleration)
    xy_norm = np.sqrt(x ** 2 + y ** 2)
    if xy_norm < 1e-10:
        xy_norm = 1e-10  # Avoid division by zero at poles

    a = np.array([
        (1 / r * dudr - z / r ** 2 / xy_norm * dudphi) * x - y / (x ** 2 + y ** 2) * dudlam,
        (1 / r * dudr - z / r ** 2 / xy_norm * dudphi) * y + x / (x ** 2 + y ** 2) * dudlam,
        z / r * dudr + xy_norm / r ** 2 * dudphi
    ])

    return a


# %[appendix]{"version":"1.0"}
