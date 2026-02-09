"""
object - 3D spacecraft model I/O and handling

yoshimuLibrary object module
Python conversion from yMATLAB/object/
"""

from .obj_io import (
    SatelliteModel,
    read_obj,
    read_sc,
    calc_area_obj,
)

from .geometry import (
    calc_normal_obj,
    calc_local_frame,
    calc_ray_intersect,
    self_shadow,
)

from .visualization import (
    show_sc,
    draw_earth,
)

__all__ = [
    'SatelliteModel', 'read_obj', 'read_sc', 'calc_area_obj',
    'calc_normal_obj', 'calc_local_frame', 'calc_ray_intersect', 'self_shadow',
    'show_sc', 'draw_earth',
]
