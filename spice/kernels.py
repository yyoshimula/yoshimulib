"""
SPICE kernel loading utilities
Python conversion from yMATLAB/SPICE/

Note: Requires SpiceyPy package: pip install spiceypy
"""

import os
from pathlib import Path
from typing import Optional, List, Union

try:
    import spiceypy as spice
    SPICEYPY_AVAILABLE = True
except ImportError:
    SPICEYPY_AVAILABLE = False


def load_spice_kernel(kernel_dir: Optional[Union[str, Path]] = None,
                      clear_existing: bool = True) -> None:
    """
    # Load SPICE kernels

    Parameters
    ----------
    kernel_dir : str or Path, optional
        directory path for kernel files.
        Default: ~/SPICE/kernels/ or uses SPICE_KERNEL_DIR environment variable
    clear_existing : bool
        if True, clear existing kernels before loading (default: True)

    Returns
    -------
    None

    Notes
    -----
    Loads standard SPICE kernels:
    - Leap seconds kernel (naif0012.tls)
    - Earth frame kernel (earth_assoc_itrf93.tf)
    - Earth binary PCK (earth_200101_990628_predict.bpc)
    - Planetary constants (pck00010.tpc)
    - Planetary ephemeris (de421.bsp)

    References
    ----------
    NAIF SPICE Toolkit: https://naif.jpl.nasa.gov/naif/

    Revisions
    ---------
    NA

    See also
    --------
    SpiceyPy documentation: https://spiceypy.readthedocs.io/
    """
    if not SPICEYPY_AVAILABLE:
        raise ImportError(
            "SpiceyPy is required for SPICE functionality. "
            "Install with: pip install spiceypy"
        )

    # Determine kernel directory
    if kernel_dir is None:
        # Check environment variable first
        kernel_dir = os.environ.get('SPICE_KERNEL_DIR')
        if kernel_dir is None:
            # Default location
            kernel_dir = Path.home() / 'SPICE' / 'kernels'
    else:
        kernel_dir = Path(kernel_dir)

    if not kernel_dir.exists():
        raise FileNotFoundError(
            f"Kernel directory not found: {kernel_dir}\n"
            f"Set SPICE_KERNEL_DIR environment variable or provide kernel_dir parameter."
        )

    # Clear existing kernels if requested
    if clear_existing:
        spice.kclear()

    # Standard kernels to load
    kernels = [
        'earth_assoc_itrf93.tf',     # Earth frame kernel
        'naif0012.tls',               # Leap seconds
        'earth_200101_990628_predict.bpc',  # Earth binary PCK
        'pck00010.tpc',               # Planetary constants
        'de421.bsp',                  # Planetary ephemeris
    ]

    # Load each kernel
    for kernel in kernels:
        kernel_path = kernel_dir / kernel
        if kernel_path.exists():
            spice.furnsh(str(kernel_path))
        else:
            print(f"Warning: Kernel not found: {kernel_path}")


def furnsh(kernel_path: Union[str, Path]) -> None:
    """
    # Load a single SPICE kernel file

    Parameters
    ----------
    kernel_path : str or Path
        path to the kernel file

    Returns
    -------
    None

    Notes
    -----
    Wrapper for spiceypy.furnsh

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    load_spice_kernel
    """
    if not SPICEYPY_AVAILABLE:
        raise ImportError(
            "SpiceyPy is required for SPICE functionality. "
            "Install with: pip install spiceypy"
        )

    kernel_path = Path(kernel_path)
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

    spice.furnsh(str(kernel_path))


def kclear() -> None:
    """
    # Clear all loaded SPICE kernels

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Wrapper for spiceypy.kclear

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    load_spice_kernel
    """
    if not SPICEYPY_AVAILABLE:
        raise ImportError(
            "SpiceyPy is required for SPICE functionality. "
            "Install with: pip install spiceypy"
        )

    spice.kclear()


def ktotal(kind: str = 'ALL') -> int:
    """
    # Return the number of loaded kernels

    Parameters
    ----------
    kind : str
        kernel type: 'ALL', 'SPK', 'CK', 'PCK', 'EK', 'TEXT', 'META'

    Returns
    -------
    count : int
        number of loaded kernels of the specified type

    Notes
    -----
    Wrapper for spiceypy.ktotal

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    load_spice_kernel
    """
    if not SPICEYPY_AVAILABLE:
        raise ImportError(
            "SpiceyPy is required for SPICE functionality. "
            "Install with: pip install spiceypy"
        )

    return spice.ktotal(kind)


def list_loaded_kernels() -> List[str]:
    """
    # List all loaded kernel files

    Parameters
    ----------
    None

    Returns
    -------
    kernels : list of str
        list of loaded kernel file paths

    Notes
    -----
    Iterates through loaded kernels using ktotal and kdata

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    load_spice_kernel
    """
    if not SPICEYPY_AVAILABLE:
        raise ImportError(
            "SpiceyPy is required for SPICE functionality. "
            "Install with: pip install spiceypy"
        )

    kernels = []
    count = spice.ktotal('ALL')

    for i in range(count):
        file, ftype, source, handle = spice.kdata(i, 'ALL')
        kernels.append(file)

    return kernels


# %[appendix]{"version":"1.0"}
