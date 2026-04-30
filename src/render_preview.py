import os
import tempfile
from pathlib import Path
 
import numpy as np
from PIL import Image
import pyvista as pv
import streamlit as st
 
os.environ["PYVISTA_OFF_SCREEN"] = "true"
 
# ── helpers ──────────────────────────────────────────────────────────────────
 
def normalize_mesh(mesh: pv.PolyData, target_size: float = 1.6) -> pv.PolyData:
    mesh = mesh.copy()
    center = np.array(mesh.center)
    mesh.translate(-center, inplace=True)
    max_dim = max(
        mesh.bounds[1] - mesh.bounds[0],
        mesh.bounds[3] - mesh.bounds[2],
        mesh.bounds[5] - mesh.bounds[4],
    )
    if max_dim > 0:
        mesh.scale(target_size / max_dim, inplace=True)
    return mesh
 
 
def make_wall(center, u_vec, v_vec, width, height):
    center = np.array(center, dtype=float)
    u = np.array(u_vec, dtype=float)
    v = np.array(v_vec, dtype=float)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
 
    p0 = center - (width / 2) * u - (height / 2) * v
    p1 = center + (width / 2) * u - (height / 2) * v
    p2 = center + (width / 2) * u + (height / 2) * v
    p3 = center - (width / 2) * u + (height / 2) * v
 
    wall = pv.PolyData(np.array([p0, p1, p2, p3], dtype=float), np.array([4, 0, 1, 2, 3]))
    wall.active_texture_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
    )
    return wall
 
 
def add_wall_border(plotter, wall, color="#333333", line_width=0.035):
    edges = wall.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    tubes = edges.tube(radius=line_width, n_sides=12)
    plotter.add_mesh(tubes, color=color, lighting=False)
 
 
def make_shadow_texture(mask_path: str, output_path: str, threshold: int = 245):
    img = Image.open(mask_path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    shadow_mask = (alpha > 0) & (luminance > threshold)
    out = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    out[:, :, 3] = shadow_mask.astype(np.uint8) * 255
    Image.fromarray(out).save(output_path)
 
 
def render_shadow_preview(
    stl_path: str,
    output_path: str,
    shadow_images=None,
    window_size=(1200, 900),
):
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
 
    mesh.translate(
        (
            -mesh.center[0],
            -mesh.center[1],
            wall_center_z - mesh.center[2],
        ),
        inplace=True,
    )
 
    plotter.add_mesh(mesh, color="#a3a3a3", smooth_shading=True, specular=0.15, diffuse=0.85)
 
    walls = [
        {
            "center": (-half, 0, wall_center_z),
            "u": (0, 1, 0), "v": (0, 0, 1),
            "width": room_size, "height": wall_height,
            "decal_center": (-half + eps, 0, wall_center_z),
        },
        {
            "center": (0, half, wall_center_z),
            "u": (1, 0, 0), "v": (0, 0, 1),
            "width": room_size, "height": wall_height,
            "decal_center": (0, half - eps, wall_center_z),
        },
        {
            "center": (0, 0, floor_z),
            "u": (1, 0, 0), "v": (0, 1, 0),
            "width": room_size, "height": room_size,
            "decal_center": (0, 0, floor_z + eps),
        },
    ]
 
    for wall_cfg in walls:
        wall = make_wall(
            center=wall_cfg["center"],
            u_vec=wall_cfg["u"],
            v_vec=wall_cfg["v"],
            width=wall_cfg["width"],
            height=wall_cfg["height"],
        )
        plotter.add_mesh(wall, color="#d8d8d8", ambient=0.75, diffuse=0.8, show_edges=False)
        add_wall_border(plotter, wall, color="#9D9D9D", line_width=0.015)
 
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, mask_path in enumerate(shadow_images[:3]):
            if mask_path is None:
                continue
            shadow_png = Path(tmpdir) / f"shadow_{i}.png"
            make_shadow_texture(mask_path, shadow_png)
            wall_cfg = walls[i]
            decal = make_wall(
                center=wall_cfg["decal_center"],
                u_vec=wall_cfg["u"],
                v_vec=wall_cfg["v"],
                width=wall_cfg["width"] * 0.3,
                height=wall_cfg["height"] * 0.3,
            )
            plotter.add_mesh(decal, texture=pv.read_texture(str(shadow_png)), lighting=False, show_edges=False)
 
        plotter.add_light(pv.Light(position=(3, -4, 5), intensity=0.8))
        plotter.add_light(pv.Light(position=(-4, -3, 3), intensity=0.4))
        plotter.camera_position = [(6.5, -7.5, 5.0), (0, 0, 0), (0, 0, 1)]
        plotter.camera.zoom(0.6)
        plotter.enable_anti_aliasing()
        plotter.screenshot(output_path)
 
    plotter.close()
    return output_path
 
 
# ── Streamlit UI ──────────────────────────────────────────────────────────────
 
st.set_page_config(page_title="3D Shadow Preview", layout="centered")
st.title("🗿 3D Shadow Preview Renderer")
st.markdown("Upload an STL file and up to three optional shadow/silhouette images, then click **Render**.")
 
stl_file = st.file_uploader("STL file (required)", type=["stl"])
 
col1, col2, col3 = st.columns(3)
with col1:
    shadow0 = st.file_uploader("Shadow — left wall", type=["png", "jpg", "jpeg", "webp"], key="s0")
with col2:
    shadow1 = st.file_uploader("Shadow — back wall", type=["png", "jpg", "jpeg", "webp"], key="s1")
with col3:
    shadow2 = st.file_uploader("Shadow — floor", type=["png", "jpg", "jpeg", "webp"], key="s2")
 
render_btn = st.button("Render", disabled=(stl_file is None), type="primary")
 
if render_btn and stl_file is not None:
    with st.spinner("Rendering… this may take 15–30 seconds."):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
 
            # Write STL
            stl_path = tmpdir / "model.stl"
            stl_path.write_bytes(stl_file.read())
 
            # Write shadow images
            shadow_paths = []
            for i, sf in enumerate([shadow0, shadow1, shadow2]):
                if sf is not None:
                    ext = Path(sf.name).suffix
                    p = tmpdir / f"shadow_{i}{ext}"
                    p.write_bytes(sf.read())
                    shadow_paths.append(str(p))
                else:
                    shadow_paths.append(None)
 
            output_path = tmpdir / "preview.png"
 
            try:
                render_shadow_preview(
                    stl_path=str(stl_path),
                    output_path=str(output_path),
                    shadow_images=shadow_paths,
                )
                img_bytes = output_path.read_bytes()
                st.image(img_bytes, caption="Rendered preview", use_container_width=True)
 
                st.download_button(
                    label="⬇️ Download PNG",
                    data=img_bytes,
                    file_name="shadow_preview.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(f"Rendering failed: {e}")