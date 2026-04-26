import os
import tempfile
from pathlib import Path

import streamlit as st

from run_pipeline import run_pipeline


st.set_page_config(
    page_title="Light Sculpture Generator",
    page_icon="./assets/favicon.png",
    layout="wide",
)

st.title("Multidirectional Light Sculpture Generator")

st.write(    "Upload 1-3 silhouette images, generate a shadow hull, and download the final STL.")

with st.sidebar:
    st.header("Settings")

    world_size = st.number_input(
        "World size",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

    grid = st.slider(
        "Voxel grid resolution",
        min_value=50,
        max_value=500,
        value=250,
        step=25,
    )

    image_size = st.slider(
        "Image size",
        min_value=50,
        max_value=500,
        value=250,
        step=25,
    )

    optimize_material = st.checkbox(
        "Optimize material / hollow shell",
        value=False,
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

    cols = st.columns(len(uploaded_files))

    for col, file in zip(cols, uploaded_files):
        with col:
            st.image(file, caption=file.name)

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
        log_box.code("\n".join(logs[-80:]))
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
                result = run_pipeline(
                    view_paths=input_paths,
                    world_size=world_size,
                    grid=grid,
                    image_size_value=image_size,
                    optimize_material=optimize_material,
                    log=ui_log,
                )

            progress.progress(100)
            st.success("Done!")

            st.subheader("3. Download STL")

            final_stl_path = (
                result["carved_stl_path"]
                if optimize_material and result["carved_stl_path"]
                else result["hull_stl_path"]
            )

            with open(final_stl_path, "rb") as f:
                st.download_button(
                    label="Download STL",
                    data=f,
                    file_name=os.path.basename(final_stl_path),
                    mime="model/stl",
                )

            st.subheader("4. Output previews")

            sim_dir = Path(result["output_dir"]) / "sim"
            if sim_dir.exists:
                sim_images = list(sim_dir.glob("*.png"))

                if sim_images:
                    cols = st.columns(min(3, len(sim_images)))

                    for i, img_path in enumerate(sim_images):
                        with cols[i % len(cols)]:
                            st.image(str(img_path), caption=img_path.name)

        except Exception as e:
            st.error("Pipeline failed.")
            st.exception(e)