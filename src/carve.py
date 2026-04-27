import numpy as np
from scipy.ndimage import distance_transform_edt, label
from projections import project_points_orthographic


def precompute_voxel_projections(voxels, voxel_centers, sources):
    occupied_coords = np.argwhere(voxels)
    pts = voxel_centers[voxels].reshape(-1, 3)

    projections = []
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

        projections.append({
            "valid": valid.copy(),
            "px": pxi,
            "py": pyi,
        })

    return occupied_coords, projections


def initialize_support_counts(projections, sources):
    counts = []
    for proj, src in zip(projections, sources):
        H, W = src.image.shape
        c = np.zeros((H, W), dtype=np.int32)

        valid = proj["valid"]
        px = proj["px"][valid]
        py = proj["py"][valid]

        np.add.at(c, (py, px), 1)
        counts.append(c)

    return counts


def compute_protected_shell(original_voxels, shell_thickness_voxels=3, protect_endcaps=True):
    dist_inside = distance_transform_edt(original_voxels)
    protected = original_voxels & (dist_inside <= shell_thickness_voxels)

    if protect_endcaps:
        coords = np.argwhere(original_voxels)
        if len(coords) > 0:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)

            x0, y0, z0 = mins
            x1, y1, z1 = maxs

            endcaps = np.zeros_like(original_voxels, dtype=bool)
            endcaps[x0, :, :] |= original_voxels[x0, :, :]
            endcaps[x1, :, :] |= original_voxels[x1, :, :]
            endcaps[:, y0, :] |= original_voxels[:, y0, :]
            endcaps[:, y1, :] |= original_voxels[:, y1, :]
            endcaps[:, :, z0] |= original_voxels[:, :, z0]
            endcaps[:, :, z1] |= original_voxels[:, :, z1]

            protected |= endcaps

    return protected


def remove_small_components(voxels, min_component_size=100):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, ncomp = label(voxels, structure=structure)

    if ncomp == 0:
        return voxels

    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(voxels, dtype=bool)

    for comp_id in range(1, len(counts)):
        if counts[comp_id] >= min_component_size:
            keep |= (labeled == comp_id)

    return keep


def carve_hollow_shell_strict(
    voxels,
    voxel_centers,
    sources,
    shell_thickness_voxels=3,
    max_passes=10,
    random_seed=0,
    protect_endcaps=True,
    cleanup_components=True,
    min_component_size=100,
    verbose=True
):
    rng = np.random.default_rng(random_seed)
    original = voxels.copy()
    carved = voxels.copy()

    protected_shell = compute_protected_shell(
        original,
        shell_thickness_voxels=shell_thickness_voxels,
        protect_endcaps=protect_endcaps
    )

    stats = {
        "start_voxels": int(carved.sum()),
        "removed": 0,
        "passes": 0,
        "shell_thickness_voxels": int(shell_thickness_voxels),
    }

    occupied_coords, projections = precompute_voxel_projections(original, voxel_centers, sources)
    counts = initialize_support_counts(projections, sources)

    active = np.ones(len(occupied_coords), dtype=bool)

    for idx, (x, y, z) in enumerate(occupied_coords):
        if not carved[x, y, z]:
            active[idx] = False

    if verbose:
        print(f"[HOLLOW] start voxels: {stats['start_voxels']}")

    for p in range(max_passes):
        stats["passes"] += 1
        removed_this_pass = 0

        candidate_indices = []
        for idx, (x, y, z) in enumerate(occupied_coords):
            if not active[idx]:
                continue
            if protected_shell[x, y, z]:
                continue
            candidate_indices.append(idx)

        candidate_indices = np.array(candidate_indices, dtype=int)

        if len(candidate_indices) == 0:
            if verbose:
                print(f"[HOLLOW] pass {p+1}: no hollowing candidates left")
            break

        rng.shuffle(candidate_indices)

        for idx in candidate_indices:
            if not active[idx]:
                continue

            x, y, z = occupied_coords[idx]

            if protected_shell[x, y, z]:
                continue

            removable = True

            for proj, src, c in zip(projections, sources, counts):
                if not proj["valid"][idx]:
                    removable = False
                    break

                xpix = proj["px"][idx]
                ypix = proj["py"][idx]

                if src.image[ypix, xpix] and c[ypix, xpix] <= 1:
                    removable = False
                    break

            if removable:
                carved[x, y, z] = False
                active[idx] = False
                removed_this_pass += 1
                stats["removed"] += 1

                for proj, c in zip(projections, counts):
                    if proj["valid"][idx]:
                        xpix = proj["px"][idx]
                        ypix = proj["py"][idx]
                        c[ypix, xpix] -= 1

        if removed_this_pass == 0:
            break

    if cleanup_components:
        before_cleanup = int(carved.sum())
        carved = remove_small_components(carved, min_component_size=min_component_size)
        after_cleanup = int(carved.sum())
        if verbose and before_cleanup != after_cleanup:
            print(f"[HOLLOW] removed {before_cleanup - after_cleanup} voxels")

    stats["end_voxels"] = int(carved.sum())
    stats["reduction_ratio"] = (
        0.0 if stats["start_voxels"] == 0
        else 1.0 - stats["end_voxels"] / stats["start_voxels"]
    )

    return carved, stats