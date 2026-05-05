import numpy as np
from projections import project_points_orthographic

def render_shadow(voxels, voxel_centers, src):
    # pixel is True if any occupied voxel projects to it
    pts = voxel_centers[voxels].reshape(-1, 3)
    H, W = src.image.shape
    out = np.zeros((H, W), dtype=bool)

    if len(pts) == 0:
        return out

    px, py, valid, _ = project_points_orthographic(
        pts,
        src.direction,
        src.up,
        src.world_center,
        src.world_size,
        src.image.shape
    )

    pxi = np.clip(np.round(px).astype(int), 0, W - 1)
    pyi = np.clip(np.round(py).astype(int), 0, H - 1)

    out[pyi[valid], pxi[valid]] = True
    return out