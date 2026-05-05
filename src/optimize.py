import numpy as np
from projections import project_points_orthographic

def inconsistent_pixels(desired, actual):
    return desired & (~actual)

def get_ray_voxel_indices_for_pixel(src, voxel_centers, pixel_xy):
    nx, ny, nz, _ = voxel_centers.shape
    pts = voxel_centers.reshape(-1, 3)

    px, py, valid, depth = project_points_orthographic(
        pts, src.direction, src.up,
        src.world_center, src.world_size, src.image.shape
    )

    xpix, ypix = pixel_xy
    pxi = np.round(px).astype(int)
    pyi = np.round(py).astype(int)

    hit = valid & (pxi == xpix) & (pyi == ypix)
    idx = np.where(hit)[0]

    idx = idx[np.argsort(depth[idx])]
    return idx

def candidate_voxel_cost(
    point3d,
    source_index,
    sources,
    outside_cost_maps
):
    # cost = sum of outside distances in all OTHER images
    total = 0.0
    for k, src in enumerate(sources):
        if k == source_index:
            continue
        px, py, valid, _ = project_points_orthographic(
            point3d[None, :],
            src.direction, src.up,
            src.world_center, src.world_size,
            src.image.shape
        )
        if not valid[0]:
            total += 1e6
            continue

        x = int(np.clip(round(px[0]), 0, src.image.shape[1] - 1))
        y = int(np.clip(round(py[0]), 0, src.image.shape[0] - 1))
        total += outside_cost_maps[k][y, x]
    return total

def find_least_cost_voxel_for_inconsistent_pixel(
    source_index,
    pixel_xy,
    sources,
    voxel_centers,
    outside_cost_maps
):
    pts = voxel_centers.reshape(-1, 3)
    idxs = get_ray_voxel_indices_for_pixel(sources[source_index], voxel_centers, pixel_xy)

    if len(idxs) == 0:
        return None

    best_idx = None
    best_cost = float("inf")

    for idx in idxs:
        cost = candidate_voxel_cost(
            pts[idx],
            source_index,
            sources,
            outside_cost_maps
        )
        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    return best_idx, best_cost