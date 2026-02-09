"""
OBJ/MTL file I/O and spacecraft model functions
Python conversion from yMATLAB/object/
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SatelliteModel:
    """
    Satellite model data structure

    Attributes
    ----------
    vertices : np.ndarray
        vertex positions (x, y, z), N x 3 matrix
    faces : np.ndarray
        face indices, N x 3 or N x 4 matrix
    normal : np.ndarray
        normal vectors, N x 3 matrix
    area : np.ndarray
        face areas, m^2, N x 1 vector
    pos : np.ndarray
        center of faces, m, N x 3 matrix
    Ca : np.ndarray
        coefficients for absorption, N x 1 vector
    Cd : np.ndarray
        coefficients for diffusion, N x 1 vector
    Cs : np.ndarray
        coefficients for specular reflection, N x 1 vector
    sunlit_flag : np.ndarray
        self shadow flag, 1: not shadowed, 0: shadowed
    mtl_file_name : str
        material library file name
    material_names : list
        list of material names
    material_indices : np.ndarray
        material index for each face
    """
    vertices: np.ndarray = field(default_factory=lambda: np.array([]))
    faces: np.ndarray = field(default_factory=lambda: np.array([]))
    normal: np.ndarray = field(default_factory=lambda: np.array([]))
    area: np.ndarray = field(default_factory=lambda: np.array([]))
    pos: np.ndarray = field(default_factory=lambda: np.array([]))
    Ca: np.ndarray = field(default_factory=lambda: np.array([]))
    Cd: np.ndarray = field(default_factory=lambda: np.array([]))
    Cs: np.ndarray = field(default_factory=lambda: np.array([]))
    sunlit_flag: np.ndarray = field(default_factory=lambda: np.array([]))
    mtl_file_name: str = ''
    material_names: list = field(default_factory=list)
    material_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    # BRDF parameters
    nu: np.ndarray = field(default_factory=lambda: np.array([]))  # Ashikhmin-Shirley
    nv: np.ndarray = field(default_factory=lambda: np.array([]))  # Ashikhmin-Shirley
    m_ct: np.ndarray = field(default_factory=lambda: np.array([]))  # Cook-Torrance
    # Physical properties
    moi: np.ndarray = field(default_factory=lambda: np.diag([10, 15, 20]))
    mass: float = 100.0
    # Local frame
    uu: np.ndarray = field(default_factory=lambda: np.array([]))
    uv: np.ndarray = field(default_factory=lambda: np.array([]))
    qlb: np.ndarray = field(default_factory=lambda: np.array([]))
    # Force and torque
    force: np.ndarray = field(default_factory=lambda: np.array([]))
    torque: np.ndarray = field(default_factory=lambda: np.array([]))


def read_obj(fname: str) -> SatelliteModel:
    """
    # Reading .obj file

    Parameters
    ----------
    fname : str
        wavefront obj file full path

    Returns
    -------
    obj : SatelliteModel
        satellite model data structure

    Notes
    -----
    This function parses wavefront object data. It reads the mesh vertices,
    texture coordinates, normal coordinates and face definitions.

    References
    ----------
    Based on Bernard Abayowa, Tec^Edge 11/8/07

    Revisions
    ---------
    20240729  modified by Yasuhiro Yoshimura

    See also
    --------
    show_sc, read_sc
    """
    v = []
    vn = []
    fv = []
    fvn = []
    mtl_file_name = ''
    material_names = []
    material_indices = []
    current_material_index = 0

    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            ln = parts[0]

            if ln == 'mtllib':
                mtl_file_name = parts[1] if len(parts) > 1 else ''

            elif ln == 'usemtl':
                mat_name = parts[1] if len(parts) > 1 else ''
                if mat_name not in material_names:
                    material_names.append(mat_name)
                current_material_index = material_names.index(mat_name)

            elif ln == 'v':
                v.append([float(x) for x in parts[1:4]])

            elif ln == 'vn':
                vn.append([float(x) for x in parts[1:4]])

            elif ln == 'f':
                face_v = []
                face_vn = []
                for p in parts[1:]:
                    indices = p.split('/')
                    face_v.append(int(indices[0]) - 1)  # 0-indexed
                    if len(indices) > 2 and indices[2]:
                        face_vn.append(int(indices[2]) - 1)

                # Pad to 4 vertices if needed
                while len(face_v) < 4:
                    face_v.append(-1)  # -1 for invalid index
                while len(face_vn) < 4:
                    face_vn.append(-1)

                fv.append(face_v[:4])
                fvn.append(face_vn[:4])
                material_indices.append(current_material_index)

    v = np.array(v)
    vn = np.array(vn) if vn else np.array([])
    fv = np.array(fv)
    fvn = np.array(fvn) if fvn else np.array([])

    # Calculate normals from face vertex normals
    normal = np.zeros((len(fv), 3))
    if len(vn) > 0 and len(fvn) > 0:
        for i, fn in enumerate(fvn):
            if fn[0] >= 0 and fn[0] < len(vn):
                normal[i] = vn[fn[0]]

    obj = SatelliteModel()
    obj.vertices = v
    obj.faces = fv
    obj.normal = normal
    obj.mtl_file_name = mtl_file_name
    obj.material_names = material_names
    obj.material_indices = np.array(material_indices)

    return obj


# %[appendix]{"version":"1.0"}


def calc_area_obj(sat: SatelliteModel) -> tuple[np.ndarray, np.ndarray]:
    """
    # calculating object facet area

    Parameters
    ----------
    sat : SatelliteModel
        satellite model data structure

    Returns
    -------
    area : np.ndarray
        face area, m^2, N x 1 vector
    pos : np.ndarray
        center of face, m, N x 3 matrix

    Notes
    -----
    Handles both triangular and quad meshes.

    References
    ----------
    NA

    Revisions
    ---------
    20250120  y.yoshimura - support for mixed tri/quad meshes
    20210209  y.yoshimura

    See also
    --------
    show_sc, read_sc
    """
    n_faces = sat.faces.shape[0]
    area = np.zeros(n_faces)
    pos = np.zeros((n_faces, 3))

    for i in range(n_faces):
        face = sat.faces[i]

        # Check if triangle or quad
        if face[3] < 0:  # triangle
            n_polygon = 3
        else:  # quad
            n_polygon = 4

        # Vectors for cross product
        vA = sat.vertices[face[1]] - sat.vertices[face[0]]
        vB = sat.vertices[face[2]] - sat.vertices[face[0]]

        cross_A = np.cross(vA, vB)

        if n_polygon == 3:
            pos_sum = (sat.vertices[face[0]] + sat.vertices[face[1]] +
                       sat.vertices[face[2]])
            pos[i] = pos_sum / 3.0
            area[i] = np.linalg.norm(cross_A) / 2.0

        elif n_polygon == 4:
            pos_sum = (sat.vertices[face[0]] + sat.vertices[face[1]] +
                       sat.vertices[face[2]] + sat.vertices[face[3]])
            pos[i] = pos_sum / 4.0
            area[i] = np.linalg.norm(cross_A)

    return area, pos


# %[appendix]{"version":"1.0"}


def read_sc(sat_name: str) -> SatelliteModel:
    """
    # Reading spacecraft shape data

    Parameters
    ----------
    sat_name : str
        satellite object file name, .obj file

    Returns
    -------
    sat : SatelliteModel
        satellite model data structure with surface properties

    Notes
    -----
    Reads OBJ file and optional MTL file for material properties.
    Maps MTL Kd (RGB) values to Ca, Cd, Cs respectively.

    References
    ----------
    NA

    Revisions
    ---------
    20240729  major update, using read_obj
    20210209  y.yoshimura

    See also
    --------
    show_sc
    """
    sat = read_obj(sat_name)

    # Calculate areas and positions
    sat.area, sat.pos = calc_area_obj(sat)

    n = len(sat.area)

    # Initialize reflectivity with default values
    sat.Ca = np.zeros(n)
    sat.Cd = np.ones(n) * 0.5
    sat.Cs = np.ones(n) * 0.5

    # Parse MTL file if available
    if sat.mtl_file_name:
        mtl_path = Path(sat_name).parent / sat.mtl_file_name
        if mtl_path.exists():
            materials = {}
            current_mat = ''

            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    if parts[0] == 'newmtl':
                        current_mat = parts[1] if len(parts) > 1 else ''
                        materials[current_mat] = {'Ka': [0, 0, 0], 'Kd': [0.5, 0.5, 0.5], 'Ks': [0, 0, 0]}
                    elif current_mat:
                        if parts[0] == 'Ka':
                            materials[current_mat]['Ka'] = [float(x) for x in parts[1:4]]
                        elif parts[0] == 'Kd':
                            materials[current_mat]['Kd'] = [float(x) for x in parts[1:4]]
                        elif parts[0] == 'Ks':
                            materials[current_mat]['Ks'] = [float(x) for x in parts[1:4]]

            # Assign material properties to faces
            for k, mat_name in enumerate(sat.material_names):
                if mat_name in materials:
                    props = materials[mat_name]
                    face_indices = np.where(sat.material_indices == k)[0]
                    # Map RGB -> Ca, Cd, Cs
                    sat.Ca[face_indices] = props['Kd'][0]
                    sat.Cd[face_indices] = props['Kd'][1]
                    sat.Cs[face_indices] = props['Kd'][2]

    # BRDF parameters (default values)
    sat.nu = np.ones(n) * 800  # Ashikhmin-Shirley
    sat.nv = np.ones(n) * 800  # Ashikhmin-Shirley
    sat.m_ct = np.ones(n) * 0.05  # Cook-Torrance

    # Physical properties (defaults)
    sat.moi = np.diag([10, 15, 20])
    sat.mass = 100.0

    # Flags and force/torque
    sat.sunlit_flag = np.ones(n)
    sat.force = np.zeros((n, 3))
    sat.torque = np.zeros((n, 3))

    return sat


# %[appendix]{"version":"1.0"}
