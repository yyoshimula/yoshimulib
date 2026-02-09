"""
geometric_integration - Geometric integration methods preserving SO(3) structure

yoshimuLibrary geometric_integration module
Python conversion from yMATLAB/geometricIntegration/
"""

from .gi import (
    butcher_table,
    q_exp,
    q_gi,
    axi_q_sol,
    euler_eom,
)

__all__ = [
    'butcher_table', 'q_exp', 'q_gi',
    'axi_q_sol', 'euler_eom',
]
