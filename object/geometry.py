"""
Geometry calculation functions for spacecraft models
Python conversion from yMATLAB/object/
"""

import numpy as np
from typing import Tuple
from .obj_io import SatelliteModel


def calc_normal_obj(sat: SatelliteModel) -> np.ndarray:
    """
    # Calculating object facet normal vectors

    Parameters
    ----------
    sat : SatelliteModel
        satellite model with vertices and faces

    Returns
    -------
    n : np.ndarray
        normal vectors, N x 3 matrix, unit vectors, outward positive

    Notes
    -----
    Calculates normal vectors from face vertices using cross product

    References
    ----------
    NA

    Revisions
    ---------
    20210209  y.yoshimura

    See also
    --------
    show_sc, read_sc, calc_area_obj
    """
    # Vector from index 1 to index 2, nx3 matrix
    v1 = sat.vertices[sat.faces[:, 1]] - sat.vertices[sat.faces[:, 0]]

    # Vector from index 1 to index 3, nx3 matrix
    v2 = sat.vertices[sat.faces[:, 2]] - sat.vertices[sat.faces[:, 0]]

    # Cross product
    cross_v = np.cross(v1, v2)

    # Norm of cross product
    cross_v_norm = np.linalg.norm(cross_v, axis=1, keepdims=True)

    # Avoid division by zero
    cross_v_norm = cross_v_norm + (cross_v_norm <= 1e-8) * 1e-8

    # Unit normal vector
    n = cross_v / cross_v_norm

    return n


def calc_local_frame(sat: SatelliteModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Calculating object facet's local frame and its quaternion from body-fixed frame

    Parameters
    ----------
    sat : SatelliteModel
        satellite model with vertices, faces, and normal vectors

    Returns
    -------
    uu : np.ndarray
        x-axis of local frame expressed with body-fixed frame, N x 3
    uv : np.ndarray
        y-axis of local frame expressed with body-fixed frame, N x 3
    qlb : np.ndarray
        quaternion from body-fixed to local frame, N x 4

    Notes
    -----
    Local frame is defined with x-axis along edge from vertex 1 to 2

    References
    ----------
    NA

    Revisions
    ---------
    20230828  y.yoshimura

    See also
    --------
    show_sc, read_sc, calc_area_obj
    """
    from ..attitude import triad, dcm2q

    n = sat.faces.shape[0]
    qlb = np.zeros((n, 4))

    # Vector from index 1 to index 2, nx3 matrix
    v1 = sat.vertices[sat.faces[:, 1]] - sat.vertices[sat.faces[:, 0]]

    # x-axis of local frame expressed with body-fixed frame
    uu = v1 / np.linalg.norm(v1, axis=1, keepdims=True)

    # Directional cosine matrix using triad method
    ref1 = np.tile([1, 0, 0], (n, 1))
    ref2 = np.tile([0, 0, 1], (n, 1))
    dcm = triad(uu, sat.normal, ref1, ref2)

    # Quaternion
    for i in range(n):
        qlb[i, :] = dcm2q(dcm[:, :, i], scalar=4)

    # y-axis of local frame
    uv = np.cross(sat.normal, uu)

    return uu, uv, qlb


def calc_ray_intersect(sun: np.ndarray, n_j: np.ndarray,
                       vert_j: np.ndarray, vert_i: np.ndarray) -> int:
    """
    # Calculating ray triangle intersection

    Parameters
    ----------
    sun : np.ndarray
        sun direction vector, 1 x 3
    n_j : np.ndarray
        normal vector of j-th facet, 1 x 3
    vert_j : np.ndarray
        vertices of j-th facet, 3 x 3 (each row is a vertex)
    vert_i : np.ndarray
        position of i-th facet center, 1 x 3

    Returns
    -------
    flag : int
        1: no shadow (ray does not intersect), 0: shadow (ray intersects)

    Notes
    -----
    Calculate if j-th facet makes shadow on i-th facet
    Based on ray-triangle intersection algorithm

    References
    ----------
    NA

    Revisions
    ---------
    20241125  added arguments, y.yoshimura
    20200811  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    self_shadow
    """
    sun = np.asarray(sun).flatten()
    vert_i = np.asarray(vert_i).flatten()
    n_j = np.asarray(n_j).flatten()

    d = np.dot(n_j, vert_j[0, :])

    # Calculate intersection scalar K
    denom = np.dot(n_j, sun)
    if np.abs(denom) < 1e-10:
        return 1

    K = (d - np.dot(n_j, vert_i)) / denom

    # Intersection point
    Q = vert_i + K * sun
    v3 = Q - vert_i
    inner = np.dot(v3, sun)

    # Check if intersection is within mesh
    if inner > 0.0:
        v1 = np.tile(Q, (3, 1)) - vert_j  # 3x3

        # v2 = 3x3 vector, connecting vertices sequentially
        v2 = np.array([
            vert_j[1, :] - vert_j[0, :],
            vert_j[2, :] - vert_j[1, :],
            vert_j[0, :] - vert_j[2, :]
        ])

        # Cross product
        cross_v = np.cross(v2, v1)
        D = np.array([
            np.dot(cross_v[0, :], cross_v[1, :]),
            np.dot(cross_v[0, :], cross_v[2, :])
        ])

        if D[0] >= 0.0 and D[1] >= 0.0:
            return 0
        else:
            return 1
    else:
        return 1


def self_shadow(sat: SatelliteModel, sun: np.ndarray,
                use_parallel: bool = False) -> SatelliteModel:
    """
    # Calculate self-shadowing

    Parameters
    ----------
    sat : SatelliteModel
        satellite configuration
    sun : np.ndarray
        Sun directional vector w.r.t. body-fixed frame, 1 x 3
    use_parallel : bool, optional
        use parallel computation (requires joblib), default False

    Returns
    -------
    sat : SatelliteModel
        updated satellite with sunlit_flag

    Notes
    -----
    Processes all facets for self-shadowing.
    j-th facet creates shadow on i-th facet.
    sunlit_flag: 0 means shadowed, 1 means sunlit

    References
    ----------
    NA

    Revisions
    ---------
    20241125  support for quad facets, y.yoshimura
    20200811  y.yoshimura, y.yoshimula@gmail.com

    See also
    --------
    calc_ray_intersect
    """
    sun = np.asarray(sun).flatten()
    sun = sun / np.linalg.norm(sun)

    n_faces = len(sat.faces)
    sat.sunlit_flag = np.zeros(n_faces)

    # Sunlit index
    sunlit_index = (sat.normal @ sun) > 0.0

    faces = sat.faces[sunlit_index, :]
    normal = sat.normal[sunlit_index, :]
    pos = sat.pos[sunlit_index, :]

    tmp_flag = np.ones(len(faces))

    def compute_shadow_for_face(i):
        """Compute shadow for a single face"""
        tmp_flag_i = 1
        pos_i = pos[i, :]

        for j in range(len(faces)):
            if j != i:
                vert_j = np.array([
                    sat.vertices[faces[j, 0], :],
                    sat.vertices[faces[j, 1], :],
                    sat.vertices[faces[j, 2], :]
                ])
                flag = calc_ray_intersect(sun, normal[j, :], vert_j, pos_i)
                tmp_flag_i *= flag

                # For quad facets, check second triangle
                if faces.shape[1] > 3 and faces[j, 3] >= 0:
                    vert_j = np.array([
                        sat.vertices[faces[j, 2], :],
                        sat.vertices[faces[j, 3], :],
                        sat.vertices[faces[j, 0], :]
                    ])
                    flag = calc_ray_intersect(sun, normal[j, :], vert_j, pos_i)
                    tmp_flag_i *= flag

        return tmp_flag_i

    if use_parallel:
        try:
            from joblib import Parallel, delayed
            tmp_flag = Parallel(n_jobs=-1)(
                delayed(compute_shadow_for_face)(i) for i in range(len(faces))
            )
            tmp_flag = np.array(tmp_flag)
        except ImportError:
            # Fall back to serial computation
            for i in range(len(faces)):
                tmp_flag[i] = compute_shadow_for_face(i)
    else:
        for i in range(len(faces)):
            tmp_flag[i] = compute_shadow_for_face(i)

    sat.sunlit_flag[sunlit_index] = tmp_flag

    return sat


# %[appendix]{"version":"1.0"}
