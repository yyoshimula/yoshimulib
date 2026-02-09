"""
Visualization functions for spacecraft models
Python conversion from yMATLAB/object/
"""

import numpy as np
from typing import Optional
from .obj_io import SatelliteModel


def show_sc(sat: SatelliteModel, normal: bool = False, ax=None):
    """
    # Visualizing spacecraft

    Parameters
    ----------
    sat : SatelliteModel
        satellite model data structure with vertices, faces, normal, pos
    normal : bool, optional
        draw normal vectors of facets (default: False)
    ax : matplotlib axes, optional
        existing 3D axes to plot on

    Returns
    -------
    fig : figure
        matplotlib figure object
    ax : axes
        matplotlib 3D axes object

    Notes
    -----
    Requires matplotlib for visualization.
    Uses mplot3d for 3D plotting.

    References
    ----------
    NA

    Revisions
    ---------
    20210127  y.yoshimura

    See also
    --------
    read_sc
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Create polygons for each face
    verts = []
    for face in sat.faces:
        if face[3] >= 0:  # quad
            polygon = [sat.vertices[face[0]], sat.vertices[face[1]],
                       sat.vertices[face[2]], sat.vertices[face[3]]]
        else:  # triangle
            polygon = [sat.vertices[face[0]], sat.vertices[face[1]],
                       sat.vertices[face[2]]]
        verts.append(polygon)

    # Add polygons to plot
    poly3d = Poly3DCollection(verts, facecolor=[0.7, 0.71, 0.71],
                               edgecolor='k', alpha=0.9)
    ax.add_collection3d(poly3d)

    if normal:
        # Draw normal vectors
        ax.quiver(sat.pos[:, 0], sat.pos[:, 1], sat.pos[:, 2],
                  sat.normal[:, 0], sat.normal[:, 1], sat.normal[:, 2],
                  color='b', arrow_length_ratio=0.1, label='normal')

        if hasattr(sat, 'uu') and len(sat.uu) > 0:
            # Draw x-axis of local frame
            ax.quiver(sat.pos[:, 0], sat.pos[:, 1], sat.pos[:, 2],
                      sat.uu[:, 0], sat.uu[:, 1], sat.uu[:, 2],
                      color='r', arrow_length_ratio=0.1, label='uu')

    # Set axis properties
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set equal aspect ratio
    max_range = np.array([
        sat.vertices[:, 0].max() - sat.vertices[:, 0].min(),
        sat.vertices[:, 1].max() - sat.vertices[:, 1].min(),
        sat.vertices[:, 2].max() - sat.vertices[:, 2].min()
    ]).max() / 2.0

    mid_x = (sat.vertices[:, 0].max() + sat.vertices[:, 0].min()) * 0.5
    mid_y = (sat.vertices[:, 1].max() + sat.vertices[:, 1].min()) * 0.5
    mid_z = (sat.vertices[:, 2].max() + sat.vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=42, azim=28)
    ax.grid(True)

    return fig, ax


def draw_earth(gmst: float, alpha: float = 1.0, const: Optional[object] = None,
               image_file: str = 'earth.jpg', n_panels: int = 180, ax=None):
    """
    # Visualizing Earth

    Parameters
    ----------
    gmst : float
        Greenwich Mean Sidereal Time, rad
    alpha : float, optional
        face alpha (transparency), 0-1 (default: 1.0)
    const : OrbitalConstants, optional
        orbital constants (default: uses R_EARTH = 6378.137 km)
    image_file : str, optional
        Earth texture image file path (default: 'earth.jpg')
    n_panels : int, optional
        number of panels for sphere (default: 180)
    ax : matplotlib axes, optional
        existing 3D axes to plot on

    Returns
    -------
    fig : figure
        matplotlib figure object
    ax : axes
        matplotlib 3D axes object

    Notes
    -----
    Requires matplotlib for visualization.
    For texture mapping, requires PIL/Pillow.

    References
    ----------
    NA

    Revisions
    ---------
    20220612  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    orbit_const
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    # Earth radius
    if const is not None:
        RE = const.RE
    else:
        RE = 6378.137  # km

    if ax is None:
        fig = plt.figure(facecolor='k')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('k')
    else:
        fig = ax.get_figure()

    # Create sphere
    u = np.linspace(0, 2 * np.pi, n_panels)
    v = np.linspace(0, np.pi, n_panels)
    x = RE * np.outer(np.cos(u), np.sin(v))
    y = RE * np.outer(np.sin(u), np.sin(v))
    z = RE * np.outer(np.ones(np.size(u)), np.cos(v))

    # Rotate by GMST
    cos_g = np.cos(gmst)
    sin_g = np.sin(gmst)
    x_rot = x * cos_g - y * sin_g
    y_rot = x * sin_g + y * cos_g

    # Try to load texture
    try:
        from PIL import Image
        from pathlib import Path

        img_path = Path(image_file)
        if img_path.exists():
            img = Image.open(img_path)
            cdata = np.array(img)
            # Normalize to 0-1
            cdata = cdata / 255.0

            ax.plot_surface(x_rot, y_rot, -z,
                            rstride=4, cstride=4,
                            facecolors=plt.cm.Blues(np.ones_like(x_rot) * 0.5),
                            alpha=alpha, shade=False)
        else:
            # No texture, just plot wireframe
            ax.plot_wireframe(x_rot, y_rot, z, color=[0.5, 0.5, 0.5], alpha=alpha)
    except ImportError:
        # No PIL, just plot wireframe
        ax.plot_wireframe(x_rot, y_rot, z, color=[0.5, 0.5, 0.5], alpha=alpha)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return fig, ax


# %[appendix]{"version":"1.0"}
