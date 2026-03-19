import numpy as np

from image_io import load_binary_image, save_mask
from voxel_ops import make_voxel_centers, voxel_pitch
from shadow_hull import compute_shadow_hull
from config import ShadowSource
from export_mesh import export_voxels_to_stl
from carve import carve_hollow_shell_strict
from simulate import simulate_and_save
from debug_slices import save_voxel_slices
from optimize_consistency import optimize_silhouettes
from reset_output import reset_output_dirs

# helper function to print out metrics per silhouette input view
def print_view_metrics(name, summaries):
    print(f"\n{name}")
    for i, m in enumerate(summaries):
        print(
            f"  view {i}: "
            f"IoU={m['iou']:.4f}, "
            f"target={m['target_pixels']}, "
            f"actual={m['actual_pixels']}, "
            f"missing={m['missing_pixels']}, "
            f"extra={m['extra_pixels']}"
        )

def main():
    # world and image parameters
    world_size = 1.0
    nx, ny, nz = 350, 350, 350
    image_size = (350, 350)
    optimize_material = False

    # reset the output directory for new run
    reset_output_dirs()

    # Load silhouettes as binary matrices
    img0 = load_binary_image("inputs/view2.png", size=image_size)
    img1 = load_binary_image("inputs/view4.png", size=image_size)
    img2 = load_binary_image("inputs/view3.png", size=image_size)

    print("[PIPELINE] img0 silhouette pixels:", img0.sum(), "of", img0.size)
    print("[PIPELINE] img1 silhouette pixels:", img1.sum(), "of", img1.size)
    print("[PIPELINE] img2 silhouette pixels:", img2.sum(), "of", img2.size)

    # save the masks to debug folder
    save_mask(img0, "outputs/debug/masks/base/view0_mask.png")
    save_mask(img1, "outputs/debug/masks/base/view1_mask.png")
    save_mask(img2, "outputs/debug/masks/base/view2_mask.png")

    # define the shadow sources 1 per image
    sources = [
        ShadowSource(
            image=img0,
            direction=np.array([1, 0, 0], dtype=float),
            up=np.array([0, 1, 0], dtype=float),
            world_center=np.array([0, 0, 0], dtype=float),
            world_size=world_size
        ),
        ShadowSource(
            image=img1,
            direction=np.array([0, 0, 1], dtype=float),
            up=np.array([0, 1, 0], dtype=float),
            world_center=np.array([0, 0, 0], dtype=float),
            world_size=world_size
        ),
        ShadowSource(
            image=img2,
            direction=np.array([0, 1, 0], dtype=float),
            up=np.array([0, 0, 1], dtype=float),
            world_center=np.array([0, 0, 0], dtype=float),
            world_size=world_size
        ),
    ]

    # print the shadow sources for debug
    for i, src in enumerate(sources):
        print(f"[PIPELINE] shadow source {i}: direction={src.direction}, up={src.up}")

    voxel_centers = make_voxel_centers(nx, ny, nz, world_size)

    # optimize silhouettes to allow arbitrary silhouette inputs
    optimized_sources = optimize_silhouettes(
        sources,
        voxel_centers,
        iterations=6,
        alpha=0.15,
        sigma=4.0,
        sample_per_view=300,
        growth_radius=2,
        verbose=True
    )

    # save optimized silhouettes for debug
    for i, src in enumerate(optimized_sources):
        save_mask(src.image, f"outputs/debug/masks/opt/view{i}_optimized_mask.png")

    # use optimized silhouettes for hull construction
    sources = optimized_sources

    # 1) Conservative hull
    hull = compute_shadow_hull(sources, voxel_centers)
    print("[PIPELINE] initial hull voxel count:", int(hull.sum()))
    print("[PIPELINE] initial hull occupancy ratio:", float(hull.mean()))

    # simulate the shadows and save to debug
    hull_summaries = simulate_and_save(
        hull,
        voxel_centers,
        sources,
        out_dir="outputs/sim",
        prefix="hull"
    )
    print_view_metrics("Hull shadow simulation", hull_summaries)

    # export raw hull mesh as stl
    pitch = voxel_pitch(world_size, nx, ny, nz)

    try:
        mesh = export_voxels_to_stl(hull, pitch, "outputs/meshes/shadow_hull.stl")
        print("[PIPELINE] saved raw hull mesh: outputs/meshes/shadow_hull.stl")
    except Exception as e:
        print("[ERROR] Raw hull export failed:", e)

    # 2) material optimization by greedy carving only if flag is true
    if optimize_material:
        # carve out the inside of the solid
        carved, carve_stats = carve_hollow_shell_strict(
            hull,
            voxel_centers,
            sources,
            shell_thickness_voxels=3,
            max_passes=1,
            random_seed=0,
            protect_endcaps=True,
            cleanup_components=True,
            min_component_size=150,
            verbose=True
        )

        print("[PIPELINE] carved sculpture voxels:", int(carved.sum()))
        print("[PIPELINE] carved sculpture occupancy ratio:", float(carved.mean()))
        print("[PIPELINE] carved sculpture voxels removed:", carve_stats["removed"])
        print("[PIPELINE] carved sculpture reduction ratio:", carve_stats["reduction_ratio"])

        # simulate the shadows of the carved sculpture and save
        carved_summaries = simulate_and_save(
            carved,
            voxel_centers,
            sources,
            out_dir="outputs/sim",
            prefix="carved"
        )
        print_view_metrics("Carved shadow simulation", carved_summaries)

        pitch = voxel_pitch(world_size, nx, ny, nz)

        # save slices to confirm optimization
        save_voxel_slices(hull, "outputs/debug/slices", "hull")
        save_voxel_slices(carved, "outputs/debug/slices", "carved")

        # export carved mesh
        try:
            mesh = export_voxels_to_stl(carved, pitch, "outputs/meshes/shadow_carved.stl")
            print("[PIPELINE] saved carved mesh: outputs/meshes/shadow_carved.stl")
        except Exception as e:
            print("[ERROR] carved export failed:", e)

if __name__ == "__main__":
    main()