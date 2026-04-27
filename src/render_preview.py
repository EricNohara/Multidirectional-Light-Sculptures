import os
import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pyvista as pv


os.environ["PYVISTA_OFF_SCREEN"] = "true"


def normalize_mesh(mesh: pv.PolyData, target_size: float = 1.6) -> pv.PolyData:
    mesh = mesh.copy()

    center = np.array(mesh.center)
    mesh.translate(-center, inplace=True)

    max_dim = max(mesh.bounds[1] - mesh.bounds[0],
                  mesh.bounds[3] - mesh.bounds[2],
                  mesh.bounds[5] - mesh.bounds[4])

    if max_dim > 0:
        mesh.scale(target_size / max_dim, inplace=True)

    return mesh


def make_wall(center, u_vec, v_vec, width, height):
    center = np.array(center, dtype=float)
    u = np.array(u_vec, dtype=float)
    v = np.array(v_vec, dtype=float)

    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    p0 = center - (width / 2) * u - (height / 2) * v
    p1 = center + (width / 2) * u - (height / 2) * v
    p2 = center + (width / 2) * u + (height / 2) * v
    p3 = center - (width / 2) * u + (height / 2) * v

    points = np.array([p0, p1, p2, p3], dtype=float)
    faces = np.array([4, 0, 1, 2, 3])

    wall = pv.PolyData(points, faces)

    tcoords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    wall.active_texture_coordinates = tcoords

    return wall

def add_wall_border(plotter, wall, color="#333333", line_width=0.035):
    edges = wall.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    tubes = edges.tube(radius=line_width, n_sides=12)

    plotter.add_mesh(
        tubes,
        color=color,
        lighting=False,
    )


def make_shadow_texture(mask_path: str, output_path: str, threshold: int = 245):
    """
    Converts a silhouette image into a transparent texture:
    - white / bright pixels become black shadow
    - dark / transparent pixels become transparent
    """
    img = Image.open(mask_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    luminance = (
        0.299 * rgb[:, :, 0] +
        0.587 * rgb[:, :, 1] +
        0.114 * rgb[:, :, 2]
    )

    shadow_mask = (alpha > 0) & (luminance > threshold)

    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)

    # black shadow
    out[:, :, :3] = 0

    # strong opacity
    out[:, :, 3] = shadow_mask.astype(np.uint8) * 255

    Image.fromarray(out).save(output_path)


def render_shadow_preview(
    stl_path: str,
    output_path: str,
    shadow_images=None,
    window_size=(1200, 900),
):
    """
    stl_path:
        Path to generated STL.

    output_path:
        Path where preview render should be saved.

    shadow_images:
        Optional list of 3 silhouette/mask images.
        These are placed as fake shadow decals on the 3 walls.
    """

    stl_path = str(stl_path)
    output_path = str(output_path)

    if shadow_images is None:
        shadow_images = []

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#393939")

    mesh = pv.read(stl_path)
    mesh = normalize_mesh(mesh, target_size=1.6)

    room_size = 5.0
    wall_height = 5.0
    half = room_size / 2.0
    eps = 0.01

    floor_z = -1.2
    wall_center_z = floor_z + wall_height / 2.0

    # Put sculpture in the center of the room, floating vertically
    target_center_x = 0.0
    target_center_y = 0.0
    target_center_z = wall_center_z

    mesh.translate(
        (
            target_center_x - mesh.center[0],
            target_center_y - mesh.center[1],
            target_center_z - mesh.center[2],
        ),
        inplace=True,
    )

    plotter.add_mesh(
        mesh,
        color="#a3a3a3",
        smooth_shading=True,
        specular=0.15,
        diffuse=0.85,
    )

    walls = [
        # input 1 -> left wall
        {
            "center": (-half, 0, wall_center_z),
            "u": (0, 1, 0),
            "v": (0, 0, 1),
            "width": room_size,
            "height": wall_height,
            "decal_center": (-half + eps, 0, wall_center_z),
        },

        # input 2 -> back/right wall
        {
            "center": (0, half, wall_center_z),
            "u": (1, 0, 0),
            "v": (0, 0, 1),
            "width": room_size,
            "height": wall_height,
            "decal_center": (0, half - eps, wall_center_z),
        },

        # input 3 -> floor
        {
            "center": (0, 0, floor_z),
            "u": (1, 0, 0),
            "v": (0, 1, 0),
            "width": room_size,
            "height": room_size,
            "decal_center": (0, 0, floor_z + eps),
        },
    ]

    # Add light gray walls/floor with visible borders
    for wall_cfg in walls:
        wall = make_wall(
            center=wall_cfg["center"],
            u_vec=wall_cfg["u"],
            v_vec=wall_cfg["v"],
            width=wall_cfg["width"],
            height=wall_cfg["height"],
        )

        plotter.add_mesh(
            wall,
            color="#d8d8d8",
            ambient=0.75,
            diffuse=0.8,
            show_edges=False,
        )

        add_wall_border(plotter, wall, color="#9D9D9D", line_width=0.015)

    # Add shadow image decals
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, mask_path in enumerate(shadow_images[:3]):
            if mask_path is None:
                continue

            shadow_png = Path(tmpdir) / f"shadow_{i}.png"
            make_shadow_texture(mask_path, shadow_png)

            wall_cfg = walls[i]

            shadow_scale = 0.3

            decal = make_wall(
                center=wall_cfg["decal_center"],
                u_vec=wall_cfg["u"],
                v_vec=wall_cfg["v"],
                width=wall_cfg["width"] * shadow_scale,
                height=wall_cfg["height"] * shadow_scale,
            )

            texture = pv.read_texture(str(shadow_png))

            plotter.add_mesh(
                decal,
                texture=texture,
                lighting=False,
                show_edges=False,
            )

        # Simple scene lighting for the object itself
        plotter.add_light(pv.Light(position=(3, -4, 5), intensity=0.8))
        plotter.add_light(pv.Light(position=(-4, -3, 3), intensity=0.4))

        plotter.camera_position = [
            (6.5, -7.5, 5.0),
            (0, 0, 0),
            (0, 0, 1),
        ]

        plotter.camera.zoom(0.6)

        plotter.enable_anti_aliasing()
        plotter.screenshot(output_path)

    plotter.close()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shadow0", default=None)
    parser.add_argument("--shadow1", default=None)
    parser.add_argument("--shadow2", default=None)

    args = parser.parse_args()

    shadows = [args.shadow0, args.shadow1, args.shadow2]
    shadows = [s for s in shadows if s is not None]

    render_shadow_preview(
        stl_path=args.stl,
        output_path=args.out,
        shadow_images=shadows,
    )