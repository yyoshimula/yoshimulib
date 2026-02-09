"""
JB2008 data reader
Python conversion from yMATLAB/orbit/readJB2008.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
from scipy.io import loadmat


def _find_file(fname: str, data_dir: Optional[str]) -> Path:
    if data_dir is not None:
        path = Path(data_dir) / fname
        if not path.exists():
            raise FileNotFoundError(f"JB2008 file not found: {path}")
        return path

    candidates = [
        Path(__file__).resolve().parent,  # orbit/
        Path(__file__).resolve().parent.parent / 'data',  # data/
        Path(__file__).resolve().parent.parent / 'yMATLAB' / 'orbit',  # yMATLAB/orbit
    ]

    for base in candidates:
        path = base / fname
        if path.exists():
            return path

    raise FileNotFoundError(
        f"JB2008 file '{fname}' not found in expected locations: {candidates}"
    )


def _read_txt_matrix(path: Path, ncols: int) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == ncols:
        return data.T
    if data.shape[0] == ncols:
        return data

    raise ValueError(
        f"Unexpected shape for {path.name}: {data.shape}, expected *x{ncols}"
    )


def read_jb2008(data_dir: Optional[str] = None
                ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    # Read Jacchia–Bowman 2008 coefficients and data (MATLAB: readJB2008)

    Parameters
    ----------
    data_dir : str, optional
        directory containing JB2008 data files

    Returns
    -------
    PC : Any
        DE430 coefficient data
    EOP : np.ndarray
        Earth orientation parameter data (13 x N)
    SOL : np.ndarray
        Space weather data (11 x N)
    DTC : np.ndarray
        Geomagnetic storm DTC data (26 x N)
    """
    coeff_path = _find_file('DE430Coeff.mat', data_dir)
    eop_path = _find_file('eop19620101.txt', data_dir)
    sol_path = _find_file('SOLFSMY.txt', data_dir)
    dtc_path = _find_file('DTCFILE.txt', data_dir)

    mat = loadmat(coeff_path)
    if 'DE430Coeff' in mat:
        pc = mat['DE430Coeff']
    else:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        pc = mat[keys[0]] if len(keys) == 1 else mat

    eop = _read_txt_matrix(eop_path, 13)
    sol = _read_txt_matrix(sol_path, 11)
    dtc = _read_txt_matrix(dtc_path, 26)

    return pc, eop, sol, dtc


def readJB2008(data_dir: Optional[str] = None
               ) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    # MATLAB-compatible alias of read_jb2008
    """
    return read_jb2008(data_dir=data_dir)

