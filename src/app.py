import os
import tempfile
from io import BytesIO
from pathlib import Path
import streamlit as st
from run_pipeline import run_pipeline
from phylopic_api import PhyloPicClient, PhyloPicImage
import numpy as np
import contextlib
from PIL import Image
import re
from streamlit_searchbox import st_searchbox


CLOUD_SAFE_MAX_GRID = 160
CLOUD_SAFE_DEFAULT_GRID = 120
CLOUD_SAFE_DEFAULT_IMAGE_SIZE = 160
MAX_VOXELS_PER_RUN = CLOUD_SAFE_MAX_GRID ** 3


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
    import streamlit as st
    import streamlit.components.v1 as components
    from pathlib import Path
    from render_preview import render_shadow_preview_threejs

    with tempfile.TemporaryDirectory() as tmpdir:
        preview_path = Path(tmpdir) / "shadow_preview.html"
        html_path = render_shadow_preview_threejs(
            stl_path=stl_path,
            output_path=str(preview_path),
        )

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        components.html(html, width=width, height=650, scrolling=False)
        return None

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


@st.cache_resource
def get_phylopic_client():
    return PhyloPicClient()


@st.cache_data(show_spinner=False, ttl=3600)
def search_phylopic_image_dicts(query, limit=12):
    return [
        image.__dict__
        for image in get_phylopic_client().search_images(query, limit=limit)
    ]


@st.cache_data(show_spinner=False)
def download_phylopic_image(uuid, title, page_url, preview_url, download_url, license_url, contributor):
    image = PhyloPicImage(
        uuid=uuid,
        title=title,
        page_url=page_url,
        preview_url=preview_url,
        download_url=download_url,
        license_url=license_url,
        contributor=contributor,
    )
    return get_phylopic_client().download_image(image)


def make_uploaded_selection(uploaded_file):
    return {
        "name": uploaded_file.name,
        "source": "Upload",
        "bytes": uploaded_file.getbuffer(),
        "caption": uploaded_file.name,
    }


def make_phylopic_selection(image, image_bytes):
    return {
        "name": f"phylopic_{image.uuid}.png",
        "source": "PhyloPic",
        "bytes": image_bytes,
        "caption": image.title,
        "details": image,
    }


def preview_selection(selection, image_size):
    return preview_uploaded_image_return(
        BytesIO(bytes(selection["bytes"])),
        image_size=image_size,
    )


def preview_phylopic_result(image, image_size):
    image_bytes = download_phylopic_image(
        image.uuid,
        image.title,
        image.page_url,
        image.preview_url,
        image.download_url,
        image.license_url,
        image.contributor,
    )
    return preview_selection(make_phylopic_selection(image, image_bytes), image_size)


def show_selected_silhouettes(selected_images, image_size):
    if not selected_images:
        return

    cols = st.columns(len(selected_images), gap="small")
    for col, selection in zip(cols, selected_images):
        with col:
            st.image(
                preview_selection(selection, image_size),
                caption=selection["caption"],
                width=image_size,
            )

            details = selection.get("details")
            if details:
                meta = []
                if isinstance(details, dict):
                    contributor = details.get("contributor")
                    license_url = details.get("license_url")
                else:
                    contributor = details.contributor
                    license_url = details.license_url
                if contributor:
                    meta.append(f"Contributor: {contributor}")
                if license_url:
                    meta.append(f"[License]({license_url})")
                st.caption(" | ".join(meta))


def get_phylopic_selected_silhouettes():
    return st.session_state.setdefault("phylopic_selected_silhouettes", [])


def remove_phylopic_selection(uuid):
    st.session_state["phylopic_selected_silhouettes"] = [
        selection
        for selection in get_phylopic_selected_silhouettes()
        if selection.get("details", {}).get("uuid") != uuid
    ]


def has_phylopic_selection(uuid):
    return any(
        selection.get("details", {}).get("uuid") == uuid
        for selection in get_phylopic_selected_silhouettes()
    )


def search_phylopic_results(query):
    if not query:
        return []

    try:
        return [
            PhyloPicImage(**image)
            for image in search_phylopic_image_dicts(query, limit=12)
        ]
    except Exception as e:
        st.error(f"PhyloPic search failed: {e}")
        return []


def add_phylopic_selection(image):
    selections = get_phylopic_selected_silhouettes()

    if len(selections) >= 3:
        st.error("Please remove a selected silhouette before adding another.")
        return
    if has_phylopic_selection(image.uuid):
        return

    try:
        image_bytes = download_phylopic_image(
            image.uuid,
            image.title,
            image.page_url,
            image.preview_url,
            image.download_url,
            image.license_url,
            image.contributor,
        )
        selection = make_phylopic_selection(image, image_bytes)
        selection["details"] = image.__dict__
        selections.append(selection)
    except Exception as e:
        st.error(f"Could not download {image.title} from PhyloPic: {e}")


def render_phylopic_result_picker(results, image_size):
    current_count = len(get_phylopic_selected_silhouettes())
    cols = st.columns(3, gap="small")

    for index, image in enumerate(results):
        with cols[index % 3]:
            try:
                preview = preview_phylopic_result(image, image_size)
            except Exception:
                preview = image.preview_url
            st.image(preview, caption=image.title, width=image_size)
            already_selected = has_phylopic_selection(image.uuid)
            disabled = already_selected or current_count >= 3
            if st.button(
                "Added" if already_selected else "Add silhouette",
                key=f"phylopic_add_{image.uuid}",
                disabled=disabled,
                type="secondary",
                use_container_width=True,
            ):
                add_phylopic_selection(image)
                st.rerun()

            if image.contributor:
                st.caption(f"Contributor: {image.contributor}")
            if image.license_url:
                st.caption(f"[License]({image.license_url})")

    if current_count >= 3:
        st.caption("Remove a selected silhouette to add another.")


def render_phylopic_selection_tray(image_size):
    selections = get_phylopic_selected_silhouettes()
    if not selections:
        return

    st.markdown("### Selected silhouettes")
    cols = st.columns(len(selections), gap="small")
    for col, selection in zip(cols, selections):
        details = selection.get("details", {})
        with col:
            st.image(
                preview_selection(selection, image_size),
                caption=selection["caption"],
                width=image_size,
            )
            if details.get("contributor"):
                st.caption(f"Contributor: {details['contributor']}")
            if details.get("license_url"):
                st.caption(f"[License]({details['license_url']})")
            if st.button(
                "Remove",
                key=f"phylopic_remove_{details.get('uuid', selection['name'])}",
                use_container_width=True,
            ):
                remove_phylopic_selection(details.get("uuid"))
                st.rerun()


def phylopic_search_ui():
    selected_taxon = st_searchbox(
        get_phylopic_client().suggest_names,
        placeholder="Type a few letters, then choose a suggested name",
        key="phylopic_searchbox",
    )

    if not selected_taxon:
        render_phylopic_selection_tray(image_size)
        return get_phylopic_selected_silhouettes()

    with st.spinner("Searching PhyloPic..."):
        results = search_phylopic_results(selected_taxon)

    if not results:
        st.info("No PhyloPic silhouettes found for that search.")
        render_phylopic_selection_tray(image_size)
        return get_phylopic_selected_silhouettes()

    st.caption(f"Showing silhouettes for {selected_taxon}.")
    render_phylopic_result_picker(results, image_size)
    render_phylopic_selection_tray(image_size)

    return get_phylopic_selected_silhouettes()


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
                            width=image_size,
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
    "Upload 1-3 silhouette images, generate a shadow hull, preview the solid, and download the final STL. You also have the option of searching through" \
    " a catalog of over 12,000 silhouettes and choosing three of your preference." \
)

st.write("Advice: Choose images that are completely dark and are similar in shape to each other in order to yield the best results.")

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
        max_value=CLOUD_SAFE_MAX_GRID,
        value=CLOUD_SAFE_DEFAULT_GRID,
        step=10,
        help="Number of voxels used to represent the volume. Higher = more detail and accuracy, but slower runtime + more memory.",
    )

    image_size = st.slider(
        "Image size",
        min_value=50,
        max_value=256,
        value=CLOUD_SAFE_DEFAULT_IMAGE_SIZE,
        step=10,
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

st.subheader("1. Choose silhouettes")

input_source = st.radio(
    "Image source",
    ["Upload files", "Search PhyloPic"],
    horizontal=True,
)

selected_images = []

if input_source == "Upload files":
    uploaded_files = st.file_uploader(
        "Upload 1 to 3 silhouette images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("Please upload at most 3 images.")
            st.stop()

        selected_images = [make_uploaded_selection(file) for file in uploaded_files]
        show_selected_silhouettes(selected_images, image_size)
else:
    selected_images = phylopic_search_ui()

if selected_images:
    st.caption(f"{len(selected_images)} silhouette(s) selected.")

st.subheader("2. Generate sculpture")

run_clicked = st.button(
    "Generate STL",
    type="primary",
    disabled=not selected_images or len(selected_images) > 3,
)

if run_clicked:
    if not selected_images:
        st.error("Please select at least one silhouette image.")
        st.stop()

    estimated_voxels = grid ** 3
    if estimated_voxels > MAX_VOXELS_PER_RUN:
        st.error(
            "This grid is too large for the hosted app. "
            f"Try {CLOUD_SAFE_MAX_GRID} or lower."
        )
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

        for i, selected_image in enumerate(selected_images):
            suffix = Path(selected_image["name"]).suffix or ".png"
            path = os.path.join(tmpdir, f"view_{i}{suffix}")

            with open(path, "wb") as f:
                f.write(bytes(selected_image["bytes"]))

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
