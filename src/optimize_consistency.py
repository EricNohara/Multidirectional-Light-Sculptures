import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing

from shadow_hull import compute_shadow_hull
from render import render_shadow
from projections import project_points_orthographic
from warp import smooth_displacement, warp_mask
from distances import outside_distance


def inconsistent_pixels(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return desired & (~actual)


def boundary_pixels(mask: np.ndarray):
    boundary = mask & (~binary_erosion(mask))
    ys, xs = np.where(boundary)
    return np.stack([xs, ys], axis=1) if len(xs) > 0 else np.zeros((0, 2), dtype=int)


def nearest_boundary_point_from_list(boundary_pts: np.ndarray, target_xy):
    if len(boundary_pts) == 0:
        return None
    diffs = boundary_pts - np.asarray(target_xy)[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    return boundary_pts[np.argmin(d2)]


def precompute_source_projection_data(sources, voxel_centers):
    """
    Project all voxel centers into every source once and build a lookup:
      (pixel_x, pixel_y) -> voxel indices along that ray, sorted by depth.
    """
    pts = voxel_centers.reshape(-1, 3)
    proj_data = []

    for src in sources:
        px, py, valid, depth = project_points_orthographic(
            pts,
            src.direction,
            src.up,
            src.world_center,
            src.world_size,
            src.image.shape
        )

        pxi = np.round(px).astype(int)
        pyi = np.round(py).astype(int)

        ray_lookup = {}
        valid_idxs = np.where(valid)[0]

        for idx in valid_idxs:
            key = (pxi[idx], pyi[idx])
            if key not in ray_lookup:
                ray_lookup[key] = []
            ray_lookup[key].append(idx)

        for key in ray_lookup:
            arr = np.array(ray_lookup[key], dtype=int)
            arr = arr[np.argsort(depth[arr])]
            ray_lookup[key] = arr

        proj_data.append({
            "px": pxi,
            "py": pyi,
            "valid": valid,
            "depth": depth,
            "ray_lookup": ray_lookup,
        })

    return pts, proj_data


def candidate_voxel_cost_fast(point_idx, source_index, sources, outside_cost_maps, proj_data):
    total = 0.0

    for k, src in enumerate(sources):
        if k == source_index:
            continue

        pd = proj_data[k]
        if not pd["valid"][point_idx]:
            total += 1e6
            continue

        x = int(np.clip(pd["px"][point_idx], 0, src.image.shape[1] - 1))
        y = int(np.clip(pd["py"][point_idx], 0, src.image.shape[0] - 1))
        total += outside_cost_maps[k][y, x]

    return total


def find_least_cost_voxel_for_inconsistent_pixel_fast(
    source_index,
    pixel_xy,
    sources,
    outside_cost_maps,
    proj_data,
    max_ray_samples=None,
):
    """
    For a missing pixel in source_index, find the least-cost voxel along that backprojected ray.
    max_ray_samples can be used to cap per-ray cost.
    """
    pd = proj_data[source_index]
    idxs = pd["ray_lookup"].get(pixel_xy, None)

    if idxs is None or len(idxs) == 0:
        return None

    if max_ray_samples is not None and len(idxs) > max_ray_samples:
        # sample uniformly along the sorted depth list
        sample_ids = np.linspace(0, len(idxs) - 1, max_ray_samples).astype(int)
        idxs = idxs[sample_ids]

    best_idx = None
    best_cost = float("inf")

    for idx in idxs:
        cost = candidate_voxel_cost_fast(
            point_idx=idx,
            source_index=source_index,
            sources=sources,
            outside_cost_maps=outside_cost_maps,
            proj_data=proj_data
        )
        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    if best_idx is None:
        return None

    return best_idx, best_cost


def project_point_to_image_fast(point_idx, src, pd):
    if not pd["valid"][point_idx]:
        return None
    x = int(np.clip(pd["px"][point_idx], 0, src.image.shape[1] - 1))
    y = int(np.clip(pd["py"][point_idx], 0, src.image.shape[0] - 1))
    return np.array([x, y], dtype=int)


def build_displacement_constraints(
    sources,
    voxel_centers,
    sample_per_view=300,
    max_ray_samples=24,
    verbose=True
):
    """
    Returns:
      dx_list, dy_list, add_maps, actuals, inconsistencies, stats
    """
    hull = compute_shadow_hull(sources, voxel_centers)
    actuals = [render_shadow(hull, voxel_centers, s) for s in sources]
    inconsistencies = [inconsistent_pixels(s.image, a) for s, a in zip(sources, actuals)]
    outside_cost_maps = [outside_distance(s.image) for s in sources]
    boundary_pts_list = [boundary_pixels(s.image) for s in sources]

    dx_list = [np.zeros_like(s.image, dtype=float) for s in sources]
    dy_list = [np.zeros_like(s.image, dtype=float) for s in sources]
    add_maps = [np.zeros_like(s.image, dtype=bool) for s in sources]

    _, proj_data = precompute_source_projection_data(sources, voxel_centers)

    total_missing = 0

    for j, inc in enumerate(inconsistencies):
        ys, xs = np.where(inc)
        total_missing += len(xs)

        if len(xs) == 0:
            continue

        candidate_ids = np.arange(len(xs))

        if len(candidate_ids) > sample_per_view:
            candidate_ids = np.random.choice(
                candidate_ids,
                size=sample_per_view,
                replace=False,
            )

        for count, t in enumerate(candidate_ids, start=1):
            px = int(xs[t])
            py = int(ys[t])

            result = find_least_cost_voxel_for_inconsistent_pixel_fast(
                source_index=j,
                pixel_xy=(px, py),
                sources=sources,
                outside_cost_maps=outside_cost_maps,
                proj_data=proj_data,
                max_ray_samples=max_ray_samples,
            )

            if result is None:
                continue

            point_idx, _ = result

            for k, src in enumerate(sources):
                if k == j:
                    continue

                q = project_point_to_image_fast(point_idx, src, proj_data[k])
                if q is None:
                    continue

                qx, qy = int(q[0]), int(q[1])

                # discrete growth suggestion
                add_maps[k][qy, qx] = True

                # smooth boundary pull
                b = nearest_boundary_point_from_list(boundary_pts_list[k], q)
                if b is not None:
                    bx, by = int(b[0]), int(b[1])
                    dx_list[k][by, bx] += (qx - bx)
                    dy_list[k][by, bx] += (qy - by)

    stats = {
        "total_missing": int(total_missing),
        "missing_per_view": [int(i.sum()) for i in inconsistencies],
        "growth_pixels_per_view": [int(a.sum()) for a in add_maps],
    }

    return dx_list, dy_list, add_maps, actuals, inconsistencies, stats


def optimize_silhouettes(
    sources,
    voxel_centers,
    iterations=8,
    alpha=0.20,
    sigma=5.0,
    sample_per_view=300,
    growth_radius=2,
    max_ray_samples=24,
    plateau_patience=2,
    fallback_growth_threshold=5,
    fallback_global_dilation=True,
    verbose=True
):
    """
    Iteratively deform silhouettes to reduce inconsistency.

    Strategy:
    - build hull from current silhouettes
    - detect missing pixels
    - infer target growth locations in the other silhouettes
    - combine:
        - smooth warp from sparse displacement field
        - discrete growth from add_maps
        - fallback dilation when progress stalls or growth is too weak
    - keep best result and early stop on plateau
    """
    current_sources = list(sources)
    best_sources = list(sources)
    best_missing = float("inf")
    stall_count = 0

    growth_structure = np.ones((2 * growth_radius + 1, 2 * growth_radius + 1), dtype=bool)
    cleanup_structure = np.ones((3, 3), dtype=bool)
    fallback_structure = np.ones((3, 3), dtype=bool)

    for it in range(iterations):
        if verbose:
            print(f"\n[OPTIMIZER] iteration {it+1}/{iterations}")

        dx_list, dy_list, add_maps, actuals, inconsistencies, stats = build_displacement_constraints(
            current_sources,
            voxel_centers,
            sample_per_view=sample_per_view,
            max_ray_samples=max_ray_samples,
            verbose=verbose
        )

        total_missing = stats["total_missing"]
        missing_per_view = stats["missing_per_view"]
        growth_pixels_per_view = stats["growth_pixels_per_view"]
        worst_view = int(np.argmax(missing_per_view)) if len(missing_per_view) > 0 else 0

        if verbose:
            print(f"[OPTIMIZER] growth pixels per view: {growth_pixels_per_view}")
            print(f"[OPTIMIZER] iter {it+1}: missing={total_missing}, per_view={missing_per_view}")

        if total_missing < best_missing:
            best_missing = total_missing
            best_sources = list(current_sources)
            stall_count = 0
        else:
            stall_count += 1

        if total_missing == 0:
            if verbose:
                print("[OPTIMIZER] perfect consistency reached")
            break

        updated_sources = []

        for idx, (src, dx, dy, add_map) in enumerate(zip(current_sources, dx_list, dy_list, add_maps)):
            # smooth sparse displacement field
            dx_s, dy_s = smooth_displacement(dx, dy, sigma=sigma)

            # continuous warp
            warped = warp_mask(
                src.image,
                dx=alpha * dx_s,
                dy=alpha * dy_s
            )

            # discrete growth proposed by least-cost-voxel constraints
            grown = binary_dilation(add_map, structure=growth_structure)

            # combine old shape + warp + targeted growth
            new_img = src.image | warped | grown

            # fallback: if growth suggestions are too weak, help the worst-view silhouette expand a bit
            if idx == worst_view and growth_pixels_per_view[idx] < fallback_growth_threshold:
                new_img = binary_dilation(new_img, structure=fallback_structure)

            # fallback on plateau: if stalled, allow a tiny additional dilation on the worst view
            if fallback_global_dilation and stall_count >= 1 and idx == worst_view:
                new_img = binary_dilation(new_img, structure=fallback_structure)

            # cleanup
            new_img = binary_closing(new_img, structure=cleanup_structure)

            updated_sources.append(type(src)(
                image=new_img,
                direction=src.direction,
                up=src.up,
                world_center=src.world_center,
                world_size=src.world_size
            ))

        current_sources = updated_sources

        if stall_count >= plateau_patience:
            if verbose:
                print("[OPTIMIZER] early stop: no improvement")
            break

    if verbose:
        print(f"[OPTIMIZER] best total missing: {best_missing}")

    return best_sources