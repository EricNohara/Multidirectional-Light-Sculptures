import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree

from shadow_hull import compute_shadow_hull
from render import render_shadow
from projections import project_points_orthographic
from image_io import save_mask


# ---------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------

def inconsistent_pixels(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return desired & (~actual)


def thin_inconsistent_layer(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    inc = inconsistent_pixels(desired, actual)
    if not inc.any():
        return inc
    return inc & (~binary_erosion(inc))


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    boundary = mask & (~binary_erosion(mask))
    ys, xs = np.where(boundary)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.stack([xs, ys], axis=1)


def nearest_boundary_point_from_list(boundary_pts: np.ndarray, target_xy) -> Optional[np.ndarray]:
    if len(boundary_pts) == 0:
        return None
    diffs = boundary_pts - np.asarray(target_xy)[None, :]
    d2 = np.sum(diffs * diffs, axis=1)
    return boundary_pts[np.argmin(d2)]


def outside_distance(mask: np.ndarray) -> np.ndarray:
    """
    Approximate distance-to-silhouette cost map for pixels outside the mask.
    Lower is better. Inside pixels get zero cost.
    """
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~mask).astype(float)


# ---------------------------------------------------------------------
# projection / least-cost voxel logic
# ---------------------------------------------------------------------

def precompute_source_projection_data(sources, voxel_centers):
    """
    Project all voxel centers into every source once and build:
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
    pd = proj_data[source_index]
    idxs = pd["ray_lookup"].get(pixel_xy, None)

    if idxs is None or len(idxs) == 0:
        return None

    if max_ray_samples is not None and len(idxs) > max_ray_samples:
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
            proj_data=proj_data,
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


# ---------------------------------------------------------------------
# mesh deformation model
# ---------------------------------------------------------------------

@dataclass
class LocalCoord:
    a: int
    b: int
    t: int
    x: float
    y: float


class ARAPModel2D:
    """
    Paper-style two-step 2D mesh deformation:
      step 1: scale-free construction
      step 2: rigid triangle fitting + edge-based reconstruction

    This is not the exact interactive precompiled implementation from Igarashi,
    but it follows the same optimization structure.
    """

    def __init__(self, rest_mask: np.ndarray, mesh_spacing: int = 16):
        self.rest_mask = rest_mask.astype(bool)
        self.h, self.w = rest_mask.shape
        self.mesh_spacing = mesh_spacing

        self.V0, self.F, self.grid_shape = self._build_regular_grid_mesh(self.w, self.h, mesh_spacing)
        self.current_vertices = self.V0.copy()

        self.local_coords = self._precompute_local_coords(self.V0, self.F)

        # tiny anchor set to stabilize solves when handle count is very small
        self.anchor_ids = self._choose_anchor_vertices()

    # -------------------------
    # mesh construction
    # -------------------------

    def _build_regular_grid_mesh(self, w: int, h: int, spacing: int):
        xs = list(range(0, w, spacing))
        ys = list(range(0, h, spacing))
        if xs[-1] != w - 1:
            xs.append(w - 1)
        if ys[-1] != h - 1:
            ys.append(h - 1)

        V = []
        for y in ys:
            for x in xs:
                V.append([float(x), float(y)])
        V = np.asarray(V, dtype=float)

        nx = len(xs)
        ny = len(ys)

        def vid(ix, iy):
            return iy * nx + ix

        F = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                v00 = vid(ix, iy)
                v10 = vid(ix + 1, iy)
                v01 = vid(ix, iy + 1)
                v11 = vid(ix + 1, iy + 1)

                # alternating diagonals helps reduce directional bias
                if (ix + iy) % 2 == 0:
                    F.append([v00, v10, v11])
                    F.append([v00, v11, v01])
                else:
                    F.append([v00, v10, v01])
                    F.append([v10, v11, v01])

        return V, np.asarray(F, dtype=int), (ny, nx)

    def _choose_anchor_vertices(self):
        ny, nx = self.grid_shape
        corners = [
            0,
            nx - 1,
            (ny - 1) * nx,
            ny * nx - 1,
        ]
        return corners

    # -------------------------
    # paper step 1 local coordinates
    # -------------------------

    def _solve_local_coords(self, pa: np.ndarray, pb: np.ndarray, pt: np.ndarray) -> Tuple[float, float]:
        e = pb - pa
        r = np.array([-e[1], e[0]], dtype=float)
        M = np.column_stack([e, r])
        rhs = pt - pa

        if abs(np.linalg.det(M)) < 1e-10:
            return 0.0, 0.0

        xy = np.linalg.solve(M, rhs)
        return float(xy[0]), float(xy[1])

    def _precompute_local_coords(self, V0: np.ndarray, F: np.ndarray) -> List[Tuple[LocalCoord, LocalCoord, LocalCoord]]:
        out = []
        for tri in F:
            i, j, k = tri.tolist()
            pi, pj, pk = V0[i], V0[j], V0[k]

            # target k from edge i->j
            xk, yk = self._solve_local_coords(pi, pj, pk)
            c_k = LocalCoord(a=i, b=j, t=k, x=xk, y=yk)

            # target i from edge j->k
            xi, yi = self._solve_local_coords(pj, pk, pi)
            c_i = LocalCoord(a=j, b=k, t=i, x=xi, y=yi)

            # target j from edge k->i
            xj, yj = self._solve_local_coords(pk, pi, pj)
            c_j = LocalCoord(a=k, b=i, t=j, x=xj, y=yj)

            out.append((c_k, c_i, c_j))
        return out

    # -------------------------
    # system builders
    # -------------------------

    def _var_x(self, vid: int) -> int:
        return 2 * vid

    def _var_y(self, vid: int) -> int:
        return 2 * vid + 1

    def _add_eq(self, rows, cols, vals, b, row_idx: int, coeffs: Dict[int, float], rhs: float):
        for c, v in coeffs.items():
            rows.append(row_idx)
            cols.append(c)
            vals.append(v)
        b.append(rhs)

    def _build_step1_system(
        self,
        handle_targets: Dict[int, np.ndarray],
        stiffness: float,
        handle_weight: float = 200.0,
        anchor_weight: float = 5.0,
    ) -> Tuple[csr_matrix, np.ndarray]:
        rows = []
        cols = []
        vals = []
        b = []
        row = 0
        n_vars = 2 * len(self.current_vertices)

        # scale-free triangle constraints from paper step 1
        for tri_local in self.local_coords:
            for c in tri_local:
                a, bb, t = c.a, c.b, c.t
                x, y = c.x, c.y

                # target_x - desired_x = 0
                coeff_x = {
                    self._var_x(t): 1.0,
                    self._var_x(a): -(1.0 - x),
                    self._var_x(bb): -x,
                    self._var_y(a): -y,
                    self._var_y(bb): +y,
                }
                self._add_eq(rows, cols, vals, b, row, coeff_x, 0.0)
                row += 1

                # target_y - desired_y = 0
                coeff_y = {
                    self._var_y(t): 1.0,
                    self._var_y(a): -(1.0 - x),
                    self._var_y(bb): -x,
                    self._var_x(a): +y,
                    self._var_x(bb): -y,
                }
                self._add_eq(rows, cols, vals, b, row, coeff_y, 0.0)
                row += 1

        # handle constraints
        for vid, tgt in handle_targets.items():
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_x(vid): handle_weight},
                handle_weight * float(tgt[0]),
            )
            row += 1
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_y(vid): handle_weight},
                handle_weight * float(tgt[1]),
            )
            row += 1

        # stiffness regularization: keep close to current shape
        if stiffness > 0:
            w = np.sqrt(stiffness)
            for vid, p in enumerate(self.current_vertices):
                self._add_eq(
                    rows, cols, vals, b, row,
                    {self._var_x(vid): w},
                    w * float(p[0]),
                )
                row += 1
                self._add_eq(
                    rows, cols, vals, b, row,
                    {self._var_y(vid): w},
                    w * float(p[1]),
                )
                row += 1

        # weak anchors for stability
        for vid in self.anchor_ids:
            p = self.current_vertices[vid]
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_x(vid): anchor_weight},
                anchor_weight * float(p[0]),
            )
            row += 1
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_y(vid): anchor_weight},
                anchor_weight * float(p[1]),
            )
            row += 1

        A = coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr()
        b = np.asarray(b, dtype=float)
        return A, b

    def _fit_triangle_rigid(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Rigid Procrustes fit of rest triangle P to intermediate triangle Q.
        Returns fitted triangle congruent to P.
        """
        cp = P.mean(axis=0)
        cq = Q.mean(axis=0)

        X = P - cp
        Y = Q - cq

        M = X.T @ Y
        U, _, Vt = np.linalg.svd(M)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        fitted = (P - cp) @ R.T + cq
        return fitted

    def _build_step2_system(
        self,
        fitted_triangles: np.ndarray,
        handle_targets: Dict[int, np.ndarray],
        stiffness: float,
        handle_weight: float = 200.0,
        anchor_weight: float = 5.0,
    ) -> Tuple[csr_matrix, np.ndarray]:
        rows = []
        cols = []
        vals = []
        b = []
        row = 0
        n_vars = 2 * len(self.current_vertices)

        # edge-based reconstruction from fitted triangles
        for tri_idx, tri in enumerate(self.F):
            fit = fitted_triangles[tri_idx]
            edges = [(0, 1), (1, 2), (2, 0)]

            for ia, ib in edges:
                va = tri[ia]
                vb = tri[ib]

                d = fit[ia] - fit[ib]

                # x edge equation
                coeff_x = {
                    self._var_x(va): 1.0,
                    self._var_x(vb): -1.0,
                }
                self._add_eq(rows, cols, vals, b, row, coeff_x, float(d[0]))
                row += 1

                # y edge equation
                coeff_y = {
                    self._var_y(va): 1.0,
                    self._var_y(vb): -1.0,
                }
                self._add_eq(rows, cols, vals, b, row, coeff_y, float(d[1]))
                row += 1

        # handle constraints
        for vid, tgt in handle_targets.items():
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_x(vid): handle_weight},
                handle_weight * float(tgt[0]),
            )
            row += 1
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_y(vid): handle_weight},
                handle_weight * float(tgt[1]),
            )
            row += 1

        # stiffness regularization
        if stiffness > 0:
            w = np.sqrt(stiffness)
            for vid, p in enumerate(self.current_vertices):
                self._add_eq(
                    rows, cols, vals, b, row,
                    {self._var_x(vid): w},
                    w * float(p[0]),
                )
                row += 1
                self._add_eq(
                    rows, cols, vals, b, row,
                    {self._var_y(vid): w},
                    w * float(p[1]),
                )
                row += 1

        # weak anchors
        for vid in self.anchor_ids:
            p = self.current_vertices[vid]
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_x(vid): anchor_weight},
                anchor_weight * float(p[0]),
            )
            row += 1
            self._add_eq(
                rows, cols, vals, b, row,
                {self._var_y(vid): anchor_weight},
                anchor_weight * float(p[1]),
            )
            row += 1

        A = coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr()
        b = np.asarray(b, dtype=float)
        return A, b

    # -------------------------
    # solver
    # -------------------------

    def solve(self, handle_targets: Dict[int, np.ndarray], stiffness: float) -> np.ndarray:
        if len(handle_targets) == 0:
            return self.current_vertices.copy()

        # step 1
        A1, b1 = self._build_step1_system(handle_targets, stiffness=stiffness)
        x1 = lsmr(A1, b1, atol=1e-6, btol=1e-6, maxiter=2000)[0]
        V1 = x1.reshape(-1, 2)

        # fit rigid triangles to intermediate result
        fitted = np.zeros((len(self.F), 3, 2), dtype=float)
        for tri_idx, tri in enumerate(self.F):
            P = self.V0[tri]
            Q = V1[tri]
            fitted[tri_idx] = self._fit_triangle_rigid(P, Q)

        # step 2
        A2, b2 = self._build_step2_system(fitted, handle_targets, stiffness=stiffness)
        x2 = lsmr(A2, b2, atol=1e-6, btol=1e-6, maxiter=2000)[0]
        V2 = x2.reshape(-1, 2)
        return V2

    # -------------------------
    # rasterization
    # -------------------------

    def rasterize_mask(self, vertices: np.ndarray) -> np.ndarray:
        """
        Warp the original rest mask from rest mesh V0 to deformed mesh vertices.
        Nearest-neighbor texture lookup of the original binary mask.
        """
        out = np.zeros((self.h, self.w), dtype=bool)

        for tri in self.F:
            src = self.V0[tri]      # rest-space triangle
            dst = vertices[tri]     # deformed-space triangle

            xmin = max(0, int(np.floor(np.min(dst[:, 0]))))
            xmax = min(self.w - 1, int(np.ceil(np.max(dst[:, 0]))))
            ymin = max(0, int(np.floor(np.min(dst[:, 1]))))
            ymax = min(self.h - 1, int(np.ceil(np.max(dst[:, 1]))))

            if xmax < xmin or ymax < ymin:
                continue

            M = np.array([
                [dst[0, 0], dst[1, 0], dst[2, 0]],
                [dst[0, 1], dst[1, 1], dst[2, 1]],
                [1.0, 1.0, 1.0]
            ], dtype=float)

            det = np.linalg.det(M)
            if abs(det) < 1e-10:
                continue

            Minv = np.linalg.inv(M)

            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    bary = Minv @ np.array([x + 0.5, y + 0.5, 1.0], dtype=float)
                    if np.all(bary >= -1e-6):
                        p_src = (
                            bary[0] * src[0]
                            + bary[1] * src[1]
                            + bary[2] * src[2]
                        )
                        sx = int(np.clip(round(float(p_src[0])), 0, self.w - 1))
                        sy = int(np.clip(round(float(p_src[1])), 0, self.h - 1))
                        if self.rest_mask[sy, sx]:
                            out[y, x] = True

        return out


# ---------------------------------------------------------------------
# constraint builder from shadow inconsistencies
# ---------------------------------------------------------------------

def build_constraints_from_inconsistencies(
    sources,
    models: List[ARAPModel2D],
    voxel_centers,
    proj_data,
    sample_per_view: int = 300,
    max_ray_samples: int = 24,
    random_seed: int = 0,
    verbose: bool = True,
) -> Tuple[List[Dict[int, np.ndarray]], dict]:
    """
    Paper-style:
    - use thin layer of inconsistent pixels in each view
    - for each selected inconsistent pixel, find least-cost voxel
    - project that voxel into the other views
    - move the closest boundary point of those other views toward the projection
    - convert those motions into mesh handle constraints
    """
    rng = np.random.default_rng(random_seed)

    hull = compute_shadow_hull(sources, voxel_centers)
    actuals = [render_shadow(hull, voxel_centers, s) for s in sources]
    thin_layers = [thin_inconsistent_layer(s.image, a) for s, a in zip(sources, actuals)]
    outside_cost_maps = [outside_distance(s.image) for s in sources]
    boundary_pts_list = [boundary_pixels(s.image) for s in sources]

    # current mesh vertex lookup in current deformed configuration
    trees = [cKDTree(m.current_vertices) for m in models]

    constraint_votes: List[Dict[int, List[np.ndarray]]] = [{} for _ in sources]

    total_missing = 0
    missing_per_view = []

    for j, inc in enumerate(thin_layers):
        ys, xs = np.where(inc)
        total_missing += len(xs)
        missing_per_view.append(int(len(xs)))

        if len(xs) == 0:
            continue

        ids = np.arange(len(xs))
        if len(ids) > sample_per_view:
            ids = rng.choice(ids, size=sample_per_view, replace=False)

        if verbose:
            print(f"[OPT] view {j}: thin inconsistent pixels={len(xs)}, sampled={len(ids)}")

        for t in ids:
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

                b = nearest_boundary_point_from_list(boundary_pts_list[k], q)
                if b is None:
                    continue

                # nearest current mesh vertex to the boundary point
                _, vid = trees[k].query(np.asarray([float(b[0]), float(b[1])]), k=1)
                vid = int(vid)

                # arrow from boundary point toward target projection
                delta = np.asarray([float(q[0] - b[0]), float(q[1] - b[1])], dtype=float)

                # move the selected handle by the same arrow
                target = models[k].current_vertices[vid] + delta

                if vid not in constraint_votes[k]:
                    constraint_votes[k][vid] = []
                constraint_votes[k][vid].append(target)

    constraints_per_view: List[Dict[int, np.ndarray]] = []
    for k in range(len(sources)):
        averaged: Dict[int, np.ndarray] = {}
        for vid, tgts in constraint_votes[k].items():
            averaged[vid] = np.mean(np.asarray(tgts, dtype=float), axis=0)
        constraints_per_view.append(averaged)

    stats = {
        "total_missing": int(total_missing),
        "missing_per_view": missing_per_view,
        "actuals": actuals,
        "thin_layers": thin_layers,
        "num_constraints_per_view": [len(c) for c in constraints_per_view],
    }

    return constraints_per_view, stats


# ---------------------------------------------------------------------
# image deformation cost / stopping
# ---------------------------------------------------------------------

def image_deformation_cost(current: np.ndarray, original: np.ndarray) -> float:
    return float(np.sum(current != original) / current.size)


def average_deformation_cost(sources, original_images) -> float:
    vals = []
    for s, o in zip(sources, original_images):
        vals.append(image_deformation_cost(s.image, o))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------
# main optimizer
# ---------------------------------------------------------------------

def optimize_silhouettes(
    sources,
    voxel_centers,
    iterations: int = 8,
    sample_per_view: int = 300,
    max_ray_samples: int = 24,
    step_fraction: float = 0.25,
    mesh_spacing: int = 16,
    stiffness_schedule: Optional[List[float]] = None,
    deformation_tolerance: float = 0.08,
    plateau_patience: int = 2,
    save_debug_masks: bool = True,
    random_seed: int = 0,
    verbose: bool = True,
):
    """
    Rewrite of the optimizer following the shadow-art paper + ARAP paper
    as closely as possible in a drop-in pipeline form.

    Key behavior:
    - thin inconsistent layers
    - least-cost voxel constraints
    - global symmetric ARAP-style deformation of ALL silhouettes
    - 2-step solve
    - partial update each iteration (default 0.25)
    - stiffness schedule
    - stop on deformation tolerance / plateau
    """
    if stiffness_schedule is None:
        stiffness_schedule = [40.0, 40.0, 15.0, 8.0, 4.0, 2.0, 1.0, 1.0]

    current_sources = list(sources)
    original_images = [s.image.copy() for s in sources]
    best_sources = list(sources)
    best_missing = float("inf")
    stall_count = 0

    # one deformation model per view; each uses its own original silhouette
    models = [ARAPModel2D(s.image.copy(), mesh_spacing=mesh_spacing) for s in sources]

    # fixed projection lookup
    _, proj_data = precompute_source_projection_data(current_sources, voxel_centers)

    for it in range(iterations):
        if verbose:
            print(f"\n[OPT] iteration {it + 1}/{iterations}")

        stiffness = stiffness_schedule[min(it, len(stiffness_schedule) - 1)]
        if verbose:
            print(f"[OPT] stiffness={stiffness:.3f}, step_fraction={step_fraction:.3f}")

        constraints_per_view, stats = build_constraints_from_inconsistencies(
            current_sources,
            models,
            voxel_centers,
            proj_data,
            sample_per_view=sample_per_view,
            max_ray_samples=max_ray_samples,
            random_seed=random_seed + it,
            verbose=verbose,
        )

        total_missing = stats["total_missing"]
        actuals = stats["actuals"]
        thin_layers = stats["thin_layers"]
        num_constraints = stats["num_constraints_per_view"]

        if verbose:
            print(f"[OPT] missing total={total_missing}, per_view={stats['missing_per_view']}")
            print(f"[OPT] constraints per view={num_constraints}")

        if total_missing < best_missing:
            best_missing = total_missing
            best_sources = list(current_sources)
            stall_count = 0
        else:
            stall_count += 1

        if total_missing == 0:
            if verbose:
                print("[OPT] perfect consistency reached")
            break

        updated_sources = []

        for idx, (src, model, constraints) in enumerate(zip(current_sources, models, constraints_per_view)):
            if len(constraints) == 0:
                updated_sources.append(type(src)(
                    image=src.image.copy(),
                    direction=src.direction,
                    up=src.up,
                    world_center=src.world_center,
                    world_size=src.world_size,
                ))
                continue

            solved_vertices = model.solve(constraints, stiffness=stiffness)

            # paper-style partial update
            stepped_vertices = (
                model.current_vertices
                + step_fraction * (solved_vertices - model.current_vertices)
            )

            new_img = model.rasterize_mask(stepped_vertices)
            model.current_vertices = stepped_vertices

            updated_sources.append(type(src)(
                image=new_img,
                direction=src.direction,
                up=src.up,
                world_center=src.world_center,
                world_size=src.world_size,
            ))

            if save_debug_masks:
                save_mask(actuals[idx], f"outputs/debug/opt_paper/iter_{it+1}_view{idx}_actual.png")
                save_mask(thin_layers[idx], f"outputs/debug/opt_paper/iter_{it+1}_view{idx}_thin_missing.png")
                save_mask(new_img, f"outputs/debug/opt_paper/iter_{it+1}_view{idx}_updated.png")

        current_sources = updated_sources

        avg_cost = average_deformation_cost(current_sources, original_images)
        if verbose:
            print(f"[OPT] avg deformation cost wrt input: {avg_cost:.6f}")

        if avg_cost > deformation_tolerance:
            if verbose:
                print("[OPT] stop: deformation tolerance exceeded")
            break

        if stall_count >= plateau_patience:
            if verbose:
                print("[OPT] stop: plateau")
            break

    if verbose:
        print(f"[OPT] best total missing: {best_missing}")

    return best_sources