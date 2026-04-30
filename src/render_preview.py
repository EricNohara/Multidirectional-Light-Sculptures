import numpy as np
import trimesh
from PIL import Image
from pathlib import Path
import tempfile
from trimesh.viewer import scene_to_html



def normalize_mesh(mesh, target_size=1.6):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)

    scale = target_size / max(mesh.extents)
    mesh.apply_scale(scale)

    return mesh


def create_wall(center, u_vec, v_vec, width, height):
    u = np.array(u_vec) / np.linalg.norm(u_vec)
    v = np.array(v_vec) / np.linalg.norm(v_vec)

    p0 = center - (width / 2) * u - (height / 2) * v
    p1 = center + (width / 2) * u - (height / 2) * v
    p2 = center + (width / 2) * u + (height / 2) * v
    p3 = center - (width / 2) * u + (height / 2) * v

    vertices = np.array([p0, p1, p2, p3])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def make_shadow_texture(mask_path, threshold=245):
    img = Image.open(mask_path).convert("RGBA")
    arr = np.array(img)

    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    luminance = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    shadow_mask = (alpha > 0) & (luminance > threshold)

    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    out[:, :, :3] = 0
    out[:, :, 3] = shadow_mask.astype(np.uint8) * 255

    return Image.fromarray(out)


def textured_plane(center, u_vec, v_vec, width, height, texture_img):
    u = np.array(u_vec) / np.linalg.norm(u_vec)
    v = np.array(v_vec) / np.linalg.norm(v_vec)

    p0 = center - (width / 2) * u - (height / 2) * v
    p1 = center + (width / 2) * u - (height / 2) * v
    p2 = center + (width / 2) * u + (height / 2) * v
    p3 = center - (width / 2) * u + (height / 2) * v

    vertices = np.array([p0, p1, p2, p3])
    faces = np.array([[0, 1, 2], [0, 2, 3]])

    uv = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture_img)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=visual,
        process=False
    )


def render_shadow_preview(
    stl_path,
    output_path,
    shadow_images=None,
):
    if shadow_images is None:
        shadow_images = []

    mesh = trimesh.load(stl_path)
    mesh = normalize_mesh(mesh)

    scene = trimesh.Scene()

    # Add main mesh
    mesh.visual.face_colors = [163, 163, 163, 255]
    scene.add_geometry(mesh)

    room_size = 5.0
    wall_height = 5.0
    half = room_size / 2
    floor_z = -1.2
    wall_center_z = floor_z + wall_height / 2
    eps = 0.01

    walls = [
        {
            "center": (-half, 0, wall_center_z),
            "u": (0, 1, 0),
            "v": (0, 0, 1),
        },
        {
            "center": (0, half, wall_center_z),
            "u": (1, 0, 0),
            "v": (0, 0, 1),
        },
        {
            "center": (0, 0, floor_z),
            "u": (1, 0, 0),
            "v": (0, 1, 0),
        },
    ]

    # Add walls
    for w in walls:
        wall = create_wall(
            np.array(w["center"]),
            w["u"],
            w["v"],
            room_size,
            wall_height if w["v"] == (0, 0, 1) else room_size,
        )
        wall.visual.face_colors = [216, 216, 216, 255]
        scene.add_geometry(wall)

    # Add shadow projections
    with tempfile.TemporaryDirectory() as tmp:
        for i, img_path in enumerate(shadow_images[:3]):
            if not img_path:
                continue

            shadow_img = make_shadow_texture(img_path)

            w = walls[i]

            center = np.array(w["center"]) + eps * np.array(w["u"])

            plane = textured_plane(
                center,
                w["u"],
                w["v"],
                room_size * 0.3,
                wall_height * 0.3 if i < 2 else room_size * 0.3,
                shadow_img,
            )

            scene.add_geometry(plane)

    # Export HTML (WebGL)
    html = scene_to_html(scene)

    html_path = output_path.replace(".png", ".html")
    with open(html_path, "w") as f:
        f.write(html)

    return html_path