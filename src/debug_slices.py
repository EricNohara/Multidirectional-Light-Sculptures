import os
import numpy as np
from PIL import Image

# Helper functions for saving slices of the mesh to debug the hollowing functionality of the pipeline
def save_slice_image(slice2d: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray((slice2d.astype(np.uint8) * 255))
    img.save(path)

def save_voxel_slices(voxels: np.ndarray, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)

    nx, ny, nz = voxels.shape

    # view the middle slices of the mesh
    sx = voxels[nx // 2, :, :]
    sy = voxels[:, ny // 2, :]
    sz = voxels[:, :, nz // 2]

    save_slice_image(sx, os.path.join(out_dir, f"{prefix}_slice_x_mid.png"))
    save_slice_image(sy, os.path.join(out_dir, f"{prefix}_slice_y_mid.png"))
    save_slice_image(sz, os.path.join(out_dir, f"{prefix}_slice_z_mid.png"))