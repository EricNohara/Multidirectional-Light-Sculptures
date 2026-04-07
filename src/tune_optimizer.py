import argparse
import csv
import itertools
import json
import os
import random
import time
from copy import deepcopy

import numpy as np

from image_io import load_binary_image
from voxel_ops import make_voxel_centers
from shadow_hull import compute_shadow_hull
from config import ShadowSource
from simulate import simulate_and_save
from optimize_consistency import optimize_silhouettes


def build_sources(images, world_size):
    """
    Same source layout as your pipeline.
    View 0: +X
    View 1: +Z
    View 2: +Y
    """
    view_configs = [
        {
            "direction": np.array([1, 0, 0], dtype=float),
            "up": np.array([0, 1, 0], dtype=float),
        },
        {
            "direction": np.array([0, 0, 1], dtype=float),
            "up": np.array([0, 1, 0], dtype=float),
        },
        {
            "direction": np.array([0, 1, 0], dtype=float),
            "up": np.array([0, 0, 1], dtype=float),
        },
    ]

    sources = []
    for i, img in enumerate(images):
        cfg = view_configs[i]
        sources.append(
            ShadowSource(
                image=img,
                direction=cfg["direction"],
                up=cfg["up"],
                world_center=np.array([0, 0, 0], dtype=float),
                world_size=world_size,
            )
        )
    return sources


def load_images(view_paths, image_size):
    images = []
    for path in view_paths:
        img = load_binary_image(path, size=(image_size, image_size))
        images.append(img)
    return images


def score_run(hull_summaries, runtime_seconds, runtime_weight=0.02):
    """
    Lower is better.

    Strongly punish missing/extra pixels.
    Lightly reward high IoU.
    Lightly penalize runtime.

    You can tweak this once you see real results.
    """
    total_missing = sum(m["missing_pixels"] for m in hull_summaries)
    total_extra = sum(m["extra_pixels"] for m in hull_summaries)
    mean_iou = sum(m["iou"] for m in hull_summaries) / len(hull_summaries)

    score = (
        10.0 * total_missing
        + 10.0 * total_extra
        + 1000.0 * (1.0 - mean_iou)
        + runtime_weight * runtime_seconds
    )
    return {
        "score": float(score),
        "total_missing": int(total_missing),
        "total_extra": int(total_extra),
        "mean_iou": float(mean_iou),
        "runtime_seconds": float(runtime_seconds),
    }


def run_single_trial(
    view_paths,
    world_size,
    grid,
    image_size,
    optimizer_params,
    sim_out_dir=None,
):
    t0 = time.time()

    images = load_images(view_paths, image_size=image_size)
    sources = build_sources(images, world_size=world_size)

    nx = ny = nz = grid
    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)

    optimized_sources = optimize_silhouettes(
        sources,
        voxel_centers,
        iterations=optimizer_params["iterations"],
        sample_per_view=optimizer_params["sample_per_view"],
        max_ray_samples=optimizer_params["max_ray_samples"],
        step_fraction=optimizer_params["step_fraction"],
        mesh_spacing=optimizer_params["mesh_spacing"],
        deformation_tolerance=optimizer_params["deformation_tolerance"],
        plateau_patience=optimizer_params["plateau_patience"],
        save_debug_masks=False,
        verbose=False,
    )

    hull = compute_shadow_hull(optimized_sources, voxel_centers)

    hull_summaries = simulate_and_save(
        hull,
        voxel_centers,
        optimized_sources,
        out_dir=sim_out_dir or "outputs/tuning_sim",
        prefix="trial",
    )

    runtime_seconds = time.time() - t0
    metrics = score_run(hull_summaries, runtime_seconds)

    result = {
        "optimizer_params": deepcopy(optimizer_params),
        "metrics": metrics,
        "per_view": hull_summaries,
    }
    return result


def grid_search_space():
    """
    Reasonable first-pass search space.
    Keep this small at first because each trial is expensive.
    """
    return {
        "iterations": [12, 20, 30],
        "sample_per_view": [200, 300, 400],
        "max_ray_samples": [16, 24, 32],
        "step_fraction": [0.20, 0.25, 0.30],
        "mesh_spacing": [12, 16],
        "deformation_tolerance": [0.02, 0.03, 0.05],
        "plateau_patience": [2, 3],
    }


def random_search_space():
    """
    Wider search space for random search.
    """
    return {
        "iterations": [10, 12, 15, 18, 20, 25, 30, 35, 40],
        "sample_per_view": [100, 150, 200, 250, 300, 350, 400, 500],
        "max_ray_samples": [8, 12, 16, 20, 24, 32, 40],
        "step_fraction": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        "mesh_spacing": [8, 10, 12, 14, 16, 20],
        "deformation_tolerance": [0.01, 0.02, 0.03, 0.05, 0.08],
        "plateau_patience": [1, 2, 3, 4],
    }


def iter_grid_configs(space):
    keys = list(space.keys())
    values = [space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def iter_random_configs(space, n_trials, seed=0):
    rng = random.Random(seed)
    keys = list(space.keys())

    seen = set()
    while len(seen) < n_trials:
        cfg = {k: rng.choice(space[k]) for k in keys}
        frozen = tuple((k, cfg[k]) for k in sorted(cfg.keys()))
        if frozen in seen:
            continue
        seen.add(frozen)
        yield cfg


def save_results_json(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_results_csv(results, out_path):
    rows = []
    for i, r in enumerate(results):
        row = {"trial": i}
        row.update(r["optimizer_params"])
        row.update(r["metrics"])
        rows.append(row)

    if not rows:
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_top_results(results, top_k=10):
    print("\n=== TOP RESULTS ===")
    for i, r in enumerate(results[:top_k], start=1):
        p = r["optimizer_params"]
        m = r["metrics"]
        print(
            f"{i:02d}. "
            f"score={m['score']:.3f}, "
            f"IoU={m['mean_iou']:.5f}, "
            f"missing={m['total_missing']}, "
            f"extra={m['total_extra']}, "
            f"runtime={m['runtime_seconds']:.2f}s | "
            f"params={p}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune optimize_silhouettes() parameters."
    )

    parser.add_argument(
        "views",
        nargs="+",
        help="1 to 3 silhouette image paths"
    )

    parser.add_argument(
        "--world-size",
        type=float,
        default=1.0,
        help="Physical size of the voxel world"
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=200,
        help="Voxel grid resolution for tuning"
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=200,
        help="Input silhouette resize resolution for tuning"
    )

    parser.add_argument(
        "--search",
        choices=["grid", "random"],
        default="random",
        help="Search strategy"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of random trials if --search=random"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best runs to print"
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/tuning",
        help="Where to save tuning results"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not (1 <= len(args.views) <= 3):
        raise ValueError("You must provide between 1 and 3 silhouette image paths.")

    if args.search == "grid":
        space = grid_search_space()
        configs = list(iter_grid_configs(space))
    else:
        space = random_search_space()
        configs = list(iter_random_configs(space, n_trials=args.trials, seed=args.seed))

    print(f"[TUNER] running {len(configs)} trials")

    results = []
    best = None

    for i, cfg in enumerate(configs, start=1):
        print(f"\n[TUNER] trial {i}/{len(configs)}")
        print(f"[TUNER] params = {cfg}")

        try:
            result = run_single_trial(
                view_paths=args.views,
                world_size=args.world_size,
                grid=args.grid,
                image_size=args.image_size,
                optimizer_params=cfg,
                sim_out_dir=os.path.join(args.out_dir, "sim"),
            )
            results.append(result)

            m = result["metrics"]
            print(
                f"[TUNER] score={m['score']:.3f}, "
                f"IoU={m['mean_iou']:.5f}, "
                f"missing={m['total_missing']}, "
                f"extra={m['total_extra']}, "
                f"runtime={m['runtime_seconds']:.2f}s"
            )

            if best is None or m["score"] < best["metrics"]["score"]:
                best = result
                print("[TUNER] new best")
        except Exception as e:
            print(f"[TUNER] trial failed: {e}")

    results.sort(key=lambda r: r["metrics"]["score"])

    json_path = os.path.join(args.out_dir, "tuning_results.json")
    csv_path = os.path.join(args.out_dir, "tuning_results.csv")

    save_results_json(results, json_path)
    save_results_csv(results, csv_path)

    print_top_results(results, top_k=args.top_k)

    if results:
        print("\n=== BEST PARAMS ===")
        print(json.dumps(results[0]["optimizer_params"], indent=2))
        print("\n=== BEST METRICS ===")
        print(json.dumps(results[0]["metrics"], indent=2))

    print(f"\n[TUNER] saved JSON: {json_path}")
    print(f"[TUNER] saved CSV:  {csv_path}")


if __name__ == "__main__":
    main()