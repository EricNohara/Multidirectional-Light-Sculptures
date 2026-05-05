import os
import numpy as np
import trimesh

def export_voxels_to_stl(voxels: np.ndarray, pitch: float, out_path: str):
    # Convert boolean voxel grid to triangle mesh with marching cubes and export STL.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if voxels.sum() == 0:
        raise ValueError("Voxel grid is empty; cannot export mesh.")

    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxels, pitch=pitch)
    mesh.export(out_path)
    return mesh