"""
Miscellaneous utility functions
Python conversion from yMATLAB/utility/
"""

from typing import Any
import sympy as sp


def sb(sym_eq: Any) -> str:
    """
    # Converting symbolic equations to latex script for Scrapbox

    Parameters
    ----------
    sym_eq : sympy expression
        symbolic equation

    Returns
    -------
    out : str
        Scrapbox-formatted LaTeX string

    Notes
    -----
    Scrapbox用にsymbolic equationを変換

    References
    ----------
    NA

    Revisions
    ---------
    20220612  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    NA
    """
    latex_str = sp.latex(sym_eq)
    out = f'[$ {latex_str}]'
    print(out)
    return out


def ifelse(condition: bool, true_value: Any, false_value: Any) -> Any:
    """
    # Ternary if-else operator

    Parameters
    ----------
    condition : bool
        condition to evaluate
    true_value : Any
        value to return if condition is True
    false_value : Any
        value to return if condition is False

    Returns
    -------
    result : Any
        true_value if condition is True, else false_value

    Notes
    -----
    Python has built-in ternary operator: x if condition else y
    This function is provided for MATLAB compatibility.

    References
    ----------
    NA

    Revisions
    ---------
    NA

    See also
    --------
    NA
    """
    return true_value if condition else false_value


# %[appendix]{"version":"1.0"}
