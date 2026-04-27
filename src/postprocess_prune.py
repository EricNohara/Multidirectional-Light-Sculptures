import numpy as np

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _reshape_voxel_centers(voxel_centers, hull_shape):
    if voxel_centers.ndim == 4 and voxel_centers.shape[:3] == hull_shape and voxel_centers.shape[3] == 3:
        return voxel_centers
    if voxel_centers.ndim == 2 and voxel_centers.shape[1] == 3 and voxel_centers.shape[0] == np.prod(hull_shape):
        return voxel_centers.reshape(*hull_shape, 3)
    raise ValueError(f"Unsupported voxel_centers shape {voxel_centers.shape} for hull shape {hull_shape}")


def _normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def _project_points_to_pixels(points, source):
    img_h, img_w = source.image.shape

    direction = _normalize(source.direction)
    up = _normalize(source.up)
    right = np.cross(direction, up)
    rn = np.linalg.norm(right)
    if rn < 1e-12:
        raise ValueError("direction and up are collinear")
    right = right / rn

    center = np.asarray(source.world_center, dtype=np.float64)
    rel = points - center[None, :]

    u = np.dot(rel, right) / float(source.world_size)
    v = np.dot(rel, up) / float(source.world_size)

    cols_f = (u + 0.5) * img_w
    rows_f = (0.5 - v) * img_h

    cols = np.floor(cols_f).astype(np.int32)
    rows = np.floor(rows_f).astype(np.int32)

    valid = (
        (rows >= 0) & (rows < img_h) &
        (cols >= 0) & (cols < img_w)
    )
    return rows, cols, valid


def _make_boundary_mask(hull):
    occ = hull.astype(bool)
    p = np.pad(occ, 1, mode="constant", constant_values=False)

    xm = p[:-2, 1:-1, 1:-1]
    xp = p[2:, 1:-1, 1:-1]
    ym = p[1:-1, :-2, 1:-1]
    yp = p[1:-1, 2:, 1:-1]
    zm = p[1:-1, 1:-1, :-2]
    zp = p[1:-1, 1:-1, 2:]

    exposed = (~xm) | (~xp) | (~ym) | (~yp) | (~zm) | (~zp)
    return occ & exposed


def _face_neighbor_count_volume(hull):
    occ = hull.astype(np.uint8)
    p = np.pad(occ, 1, mode="constant", constant_values=0)

    return (
        p[:-2, 1:-1, 1:-1] +
        p[2:, 1:-1, 1:-1] +
        p[1:-1, :-2, 1:-1] +
        p[1:-1, 2:, 1:-1] +
        p[1:-1, 1:-1, :-2] +
        p[1:-1, 1:-1, 2:]
    )


def _largest_component_only(hull):
    hull = hull.astype(bool)
    if hull.sum() == 0:
        return hull, 0, 0

    if not _HAS_SCIPY:
        # if scipy is unavailable, skip this cleanup rather than being slow
        return hull, None, 0

    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels, num = ndimage.label(hull, structure=structure)

    if num <= 1:
        return hull, int(num), 0

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep_label = int(np.argmax(counts))

    new_hull = (labels == keep_label)
    removed = int(hull.sum() - new_hull.sum())
    return new_hull, int(num), removed


def _build_projection_data(hull, voxel_centers, sources):
    centers4 = _reshape_voxel_centers(voxel_centers, hull.shape)

    occ_coords = np.argwhere(hull)
    occ_points = centers4[hull]

    voxel_to_occ_index = -np.ones(hull.shape, dtype=np.int64)
    voxel_to_occ_index[hull] = np.arange(len(occ_coords), dtype=np.int64)

    rows_per_view = []
    cols_per_view = []
    valids_per_view = []
    counts_per_view = []

    for src in sources:
        rows, cols, valid = _project_points_to_pixels(occ_points, src)
        H, W = src.image.shape

        lin = rows[valid] * W + cols[valid]
        counts = np.bincount(lin, minlength=H * W).reshape(H, W).astype(np.int32)

        rows_per_view.append(rows)
        cols_per_view.append(cols)
        valids_per_view.append(valid)
        counts_per_view.append(counts)

    return {
        "occ_coords": occ_coords,
        "voxel_to_occ_index": voxel_to_occ_index,
        "rows_per_view": rows_per_view,
        "cols_per_view": cols_per_view,
        "valids_per_view": valids_per_view,
        "counts_per_view": counts_per_view,
    }


def _rebuild_projection_data(hull, voxel_centers, sources):
    return _build_projection_data(hull, voxel_centers, sources)


def _support_stats_for_voxel(occ_i, proj_data, optimized_sources, original_sources=None):
    """
    Returns a tuple:
      breaks_optimized_required: bool
      redundancy_score: float
      original_penalty: float
    """
    redundancy = 0.0
    original_penalty = 0.0

    for v, opt_src in enumerate(optimized_sources):
        valid = proj_data["valids_per_view"][v][occ_i]
        if not valid:
            continue

        r = proj_data["rows_per_view"][v][occ_i]
        c = proj_data["cols_per_view"][v][occ_i]
        cnt = proj_data["counts_per_view"][v][r, c]

        # If this voxel is sole support for a required optimized pixel, cannot remove.
        if cnt == 1 and opt_src.image[r, c]:
            return True, 0.0, 0.0

        # reward removal if lots of support remains
        if cnt > 1:
            redundancy += min(cnt - 1, 6)

        # reward removal if optimized silhouette does not even want this pixel
        if not opt_src.image[r, c]:
            redundancy += 1.5

        if original_sources is not None:
            orig_src = original_sources[v]
            if orig_src.image[r, c]:
                # small penalty if original wanted this pixel
                original_penalty += 0.75
            else:
                # small reward if original did not want this pixel
                original_penalty -= 0.75

    return False, redundancy, original_penalty


def _apply_removals_bulk(hull, remove_mask):
    hull = hull.copy()
    hull[remove_mask] = False
    return hull


def fast_projection_prune(
    hull,
    voxel_centers,
    optimized_sources,
    original_sources=None,
    max_passes=4,
    max_remove_fraction_per_pass=0.12,
    min_face_neighbors=2,
    redundancy_threshold=3.0,
    cleanup_each_pass=True,
    verbose=True,
):
    """
    Fast postprocess:
      - boundary-only
      - bulk removal of projection-redundant voxels
      - optional largest-component cleanup after each pass

    This is designed to be much faster than exact connected pruning.
    """
    hull = hull.astype(bool).copy()

    hull, num0, removed0 = _largest_component_only(hull)
    if verbose:
        print(f"[POSTPROCESS] initial component cleanup: removed={removed0}, num_components={num0}")

    total_bulk_removed = 0
    total_cc_removed = removed0

    for p in range(max_passes):
        if hull.sum() == 0:
            break

        boundary = _make_boundary_mask(hull)
        if not boundary.any():
            if verbose:
                print(f"[POSTPROCESS] pass {p+1}: no boundary voxels")
            break

        proj_data = _rebuild_projection_data(hull, voxel_centers, optimized_sources)
        face_counts = _face_neighbor_count_volume(hull)

        candidates = np.argwhere(boundary)
        scored = []

        for x, y, z in candidates:
            occ_i = proj_data["voxel_to_occ_index"][x, y, z]
            if occ_i < 0:
                continue

            face_n = int(face_counts[x, y, z])

            # protect very structural voxels
            if face_n >= 5:
                continue

            breaks_opt, redundancy, orig_term = _support_stats_for_voxel(
                occ_i,
                proj_data,
                optimized_sources,
                original_sources=original_sources,
            )

            if breaks_opt:
                continue

            # prefer weakly attached voxels and highly redundant voxels
            score = redundancy + orig_term + (4 - min(face_n, 4)) * 1.25

            if face_n < min_face_neighbors:
                score += 1.5

            if score >= redundancy_threshold:
                scored.append((score, x, y, z))

        if not scored:
            if verbose:
                print(f"[POSTPROCESS] pass {p+1}: no removable candidates")
            break

        scored.sort(reverse=True, key=lambda t: t[0])

        max_remove = max(1, int(max_remove_fraction_per_pass * hull.sum()))
        scored = scored[:max_remove]

        remove_mask = np.zeros_like(hull, dtype=bool)
        for _, x, y, z in scored:
            remove_mask[x, y, z] = True

        before = int(hull.sum())
        hull = _apply_removals_bulk(hull, remove_mask)
        after_bulk = int(hull.sum())
        removed_bulk = before - after_bulk
        total_bulk_removed += removed_bulk

        cc_removed = 0
        num_comp = None
        if cleanup_each_pass:
            hull, num_comp, cc_removed = _largest_component_only(hull)
            total_cc_removed += cc_removed

        after = int(hull.sum())

        if verbose:
            print(
                f"[POSTPROCESS] pass {p+1}: bulk_removed={removed_bulk}, "
                f"cc_removed={cc_removed}, final_voxels={after}, num_components={num_comp}"
            )

        if removed_bulk == 0 and cc_removed == 0:
            break

    # final guarantee: keep only largest connected component
    hull, numf, removedf = _largest_component_only(hull)
    total_cc_removed += removedf

    if verbose:
        print(f"[POSTPROCESS] final component cleanup: removed={removedf}, num_components={numf}")
        print(f"[POSTPROCESS] final voxel count: {int(hull.sum())}")

    stats = {
        "bulk_removed": int(total_bulk_removed),
        "cc_removed": int(total_cc_removed),
        "final_voxels": int(hull.sum()),
    }

    return hull, stats