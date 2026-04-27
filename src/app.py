import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import tempfile
from pathlib import Path
import streamlit as st
from run_pipeline import run_pipeline
from render_preview import render_shadow_preview
import numpy as np
import contextlib
from PIL import Image
import re


class StreamlitLogCapture:
    def __init__(self, log_func):
        self.log_func = log_func
        self.buffer = ""

    def write(self, text):
        self.buffer += text

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.log_func(line)

    def flush(self):
        if self.buffer.strip():
            self.log_func(self.buffer.strip())
            self.buffer = ""


def show_stl_preview(stl_path, sim_dir=None, width=420):
    preview_path = str(Path(stl_path).with_name("shadow_preview.png"))
    shadow_images = []

    if sim_dir and Path(sim_dir).exists():
        sim_images = sorted(Path(sim_dir).glob("*.png"))
        view_map = {}

        for img in sim_images:
            match = re.search(r"view[_-]?(\d+)", img.name.lower())
            if match:
                v = int(match.group(1))
                if v not in view_map:
                    view_map[v] = img

        shadow_images = [view_map.get(i) for i in range(3)]

    render_shadow_preview(
        stl_path=stl_path,
        output_path=preview_path,
        shadow_images=[str(p) if p else None for p in shadow_images],
    )

    st.image(preview_path, caption="Generated Shadow Preview", width=width)
    return preview_path


def show_shadow_stats(hull_summaries):
    if not hull_summaries:
        st.info("No shadow stats available.")
        return

    cols = st.columns(len(hull_summaries), gap="small")

    for i, (col, m) in enumerate(zip(cols, hull_summaries)):
        with col:
            st.markdown(f"**View {i}**")
            st.markdown(
                f"""
                <div style="
                    border:1px solid #333842;
                    border-radius:10px;
                    padding:10px;
                    background:#151922;
                ">
                    <div><b>Accuracy:</b> {m['iou'] * 100:.2f}%</div>
                    <div><b>Missing:</b> {m['missing_pixels']}</div>
                    <div><b>Extra:</b> {m['extra_pixels']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def preview_uploaded_image_return(uploaded_file, image_size=250, threshold=128, invert=False):
    img = Image.open(uploaded_file).convert("RGBA").resize((image_size, image_size))

    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("L")

    arr = np.array(img)

    mask = arr < threshold
    if invert:
        mask = ~mask

    preview_arr = np.where(mask, 0, 255).astype(np.uint8)
    return Image.fromarray(preview_arr, mode="L")


def render_scrollable_logs(logs):
    html = "<br>".join(logs[-300:])

    st.markdown(
        f"""
        <div style="
            max-height: 320px;
            overflow-y: auto;
            background-color: #0e1117;
            color: #fafafa;
            padding: 0.75rem;
            border-radius: 0.5rem;
            font-family: monospace;
            font-size: 0.85rem;
            line-height: 1.35;
            white-space: pre-wrap;
        ">{html}</div>
        """,
        unsafe_allow_html=True,
    )


def get_view_number(path):
    match = re.search(r"view[_-]?(\d+)", path.name.lower())
    if match:
        return int(match.group(1))
    return 999


def show_results(result, optimize_material):
    final_stl_path = (
        result["carved_stl_path"]
        if optimize_material and result["carved_stl_path"]
        else result["hull_stl_path"]
    )

    sim_dir = Path(result["output_dir"]) / "sim"

    st.subheader("3. Preview")

    preview_col, info_col = st.columns([1.1, 1])

    with preview_col:
        show_stl_preview(
            final_stl_path,
            sim_dir=sim_dir,
            width=420,
        )

    with info_col:
        st.markdown("### Result Stats")
        show_shadow_stats(result.get("hull_summaries"))

        st.markdown("### Download")

        with open(final_stl_path, "rb") as f:
            st.download_button(
                label="Download STL",
                data=f,
                file_name=os.path.basename(final_stl_path),
                mime="model/stl",
                type="primary",
            )

    st.subheader("4. Output previews")

    if sim_dir.exists():
        sim_images = list(sim_dir.glob("*.png"))

        if sim_images:
            grouped = {}

            for img_path in sim_images:
                view_num = get_view_number(img_path)
                grouped.setdefault(view_num, []).append(img_path)

            for view_num in sorted(grouped.keys()):
                label = f"View {view_num}" if view_num != 999 else "Other"
                st.markdown(f"**{label}**")

                row_images = sorted(grouped[view_num], key=lambda p: p.name)
                cols = st.columns(len(row_images))

                for col, img_path in zip(cols, row_images):
                    with col:
                        name = img_path.name.lower()

                        if "target" in name:
                            caption = "Target Silhouette (Input)"
                        elif "actual" in name:
                            caption = "Rendered Silhouette"
                        elif "comparison" in name:
                            caption = "Comparison to Target"
                        elif "missing" in name:
                            caption = "Missing Regions (Target - Actual)"
                        elif "extra" in name:
                            caption = "Extra Regions (Actual - Target)"
                        else:
                            caption = img_path.name

                        st.image(
                            str(img_path),
                            caption=caption,
                            width=220,
                        )

    st.divider()

    if st.button("Reset", type="secondary"):
        st.session_state.clear()
        st.rerun()


st.set_page_config(
    page_title="Light Sculpture Generator",
    page_icon="./assets/favicon.png",
    layout="wide",
)

st.title("Multidirectional Light Sculpture Generator")

st.write(
    "Upload 1-3 silhouette images, generate a shadow hull, preview the solid, and download the final STL."
)

st.divider()

with st.sidebar:
    st.header("Settings")

    world_size = st.slider(
        "World size",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        help="Controls the overall size of the 3D space the sculpture is built in. Larger values allow bigger structures.",
    )

    grid = st.slider(
        "Voxel grid resolution",
        min_value=50,
        max_value=500,
        value=250,
        step=25,
        help="Number of voxels used to represent the volume. Higher = more detail and accuracy, but slower runtime + more memory.",
    )

    image_size = st.slider(
        "Image size",
        min_value=50,
        max_value=500,
        value=250,
        step=25,
        help="Resolution used when processing silhouette images. Higher = sharper projections but slower optimization.",
    )

    prune_passes = st.slider(
        "Material pruning passes",
        min_value=0,
        max_value=12,
        value=6,
        step=1,
        help="Controls how many pruning passes remove redundant voxels. Higher removes more material but may increase runtime and risk over-pruning.",
    )

    optimize_material = st.checkbox(
        "Hollow sculpture",
        value=False,
        help="Removes unnecessary interior voxels to create a hollow structure. Reduces material usage but increases processing time.",
    )

st.subheader("1. Upload silhouettes")

uploaded_files = st.file_uploader(
    "Upload 1 to 3 silhouette images",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.error("Please upload at most 3 images.")
        st.stop()

    cols = st.columns(len(uploaded_files), gap="small")

    for i, (col, file) in enumerate(zip(cols, uploaded_files)):
        img = preview_uploaded_image_return(file, image_size=image_size)

        with col:
            st.image(
                img,
                caption=file.name,
                width=260,
            )

st.subheader("2. Generate sculpture")

run_clicked = st.button(
    "Generate STL",
    type="primary",
    disabled=not uploaded_files or len(uploaded_files) > 3,
)

if run_clicked:
    if not uploaded_files:
        st.error("Please upload at least one silhouette image.")
        st.stop()

    log_box = st.empty()
    progress = st.progress(0)
    logs = []

    def ui_log(message):
        logs.append(str(message))

        with log_box.container():
            render_scrollable_logs(logs)

        progress.progress(min(95, 5 + len(logs) * 3))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_paths = []

        for i, uploaded_file in enumerate(uploaded_files):
            suffix = Path(uploaded_file.name).suffix or ".png"
            path = os.path.join(tmpdir, f"view_{i}{suffix}")

            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            input_paths.append(path)

        try:
            with st.spinner("Generating sculpture..."):
                capture = StreamlitLogCapture(ui_log)

                with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                    result = run_pipeline(
                        view_paths=input_paths,
                        world_size=world_size,
                        grid=grid,
                        image_size_value=image_size,
                        optimize_material=optimize_material,
                        prune_passes=prune_passes,
                        log=ui_log,
                    )

            progress.progress(100)
            st.success("Done!")

            st.session_state["result"] = result
            st.session_state["result_optimize_material"] = optimize_material

        except Exception as e:
            st.error("Pipeline failed.")
            st.exception(e)

if "result" in st.session_state:
    show_results(
        st.session_state["result"],
        st.session_state.get("result_optimize_material", False),
    )