import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from projections import project_points_orthographic


def compute_raw_shadow_hull(sources, voxel_centers):
    """
    Standard voxel shadow hull:
    keep a voxel only if it projects inside EVERY silhouette.
    """
    nx, ny, nz, _ = voxel_centers.shape
    pts = voxel_centers.reshape(-1, 3)

    keep = np.ones(len(pts), dtype=bool)

    for src in sources:
        px, py, valid, _ = project_points_orthographic(
            pts,
            src.direction,
            src.up,
            src.world_center,
            src.world_size,
            src.image.shape
        )

        pxi = np.clip(np.round(px).astype(int), 0, src.image.shape[1] - 1)
        pyi = np.clip(np.round(py).astype(int), 0, src.image.shape[0] - 1)

        inside = np.zeros(len(pts), dtype=bool)
        inside[valid] = src.image[pyi[valid], pxi[valid]]

        keep &= inside

    return keep.reshape(nx, ny, nz)


def connected_components_3d(mask):
    """
    6-connected components in 3D:
    only face-adjacent neighbors in +/-x, +/-y, +/-z.
    """
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True

    structure[0, 1, 1] = True  # -x
    structure[2, 1, 1] = True  # +x
    structure[1, 0, 1] = True  # -y
    structure[1, 2, 1] = True  # +y
    structure[1, 1, 0] = True  # -z
    structure[1, 1, 2] = True  # +z

    labels, num = label(mask, structure=structure)
    return labels, num


def line3d_voxels(p0, p1):
    """
    Straight voxel line between two 3D integer points.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    n = int(np.max(np.abs(p1 - p0))) + 1
    pts = np.round(np.linspace(p0, p1, n)).astype(int)

    out = []
    seen = set()
    for p in pts:
        key = tuple(p.tolist())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return np.array(out, dtype=int)


def connect_all_components_fast(hull, min_component_size=1, verbose=False):
    """
    Faster version:
    - label once
    - find largest component
    - connect every other sufficiently large component directly to it
    - use KD-tree instead of brute-force closest pair
    """
    hull = hull.copy()

    labels, num = connected_components_3d(hull)
    if num <= 1:
        return hull

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background

    main_label = int(np.argmax(counts))
    main_pts = np.argwhere(labels == main_label)

    if len(main_pts) == 0:
        return hull

    main_tree = cKDTree(main_pts)

    nx, ny, nz = hull.shape

    for lab in range(1, num + 1):
        if lab == main_label:
            continue

        comp_size = int(counts[lab])
        if comp_size < min_component_size:
            continue

        comp_pts = np.argwhere(labels == lab)
        if len(comp_pts) == 0:
            continue

        # Find closest point in main component for each point in this component
        dists, idxs = main_tree.query(comp_pts, k=1)

        best_i = int(np.argmin(dists))
        pb = comp_pts[best_i]          # point in floating component
        pa = main_pts[idxs[best_i]]    # closest point in main component

        path = line3d_voxels(pa, pb)

        if verbose:
            print(
                f"[SHADOW HULL] connecting component {lab} "
                f"(size={comp_size}) with bridge length {len(path)}"
            )

        path = path[
            (path[:, 0] >= 0) & (path[:, 0] < nx) &
            (path[:, 1] >= 0) & (path[:, 1] < ny) &
            (path[:, 2] >= 0) & (path[:, 2] < nz)
        ]

        hull[path[:, 0], path[:, 1], path[:, 2]] = True

    return hull


def compute_shadow_hull(
    sources,
    voxel_centers,
    enforce_connectivity=True,
    min_component_size=1,
    verbose=False,
):
    """
    Fast final hull:
    1. build raw shadow hull
    2. if disconnected, connect floating components to the largest component
       using simple straight voxel bridges
    """
    hull = compute_raw_shadow_hull(sources, voxel_centers)

    if not enforce_connectivity or not np.any(hull):
        return hull

    labels, num = connected_components_3d(hull)
    if num <= 1:
        return hull

    hull = connect_all_components_fast(
        hull,
        min_component_size=min_component_size,
        verbose=verbose,
    )

    return hull