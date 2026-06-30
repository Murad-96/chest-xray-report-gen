"""
Streamlit demo for the chest X-ray report generator.

Run from the repo root:
    streamlit run app/streamlit_app.py

Two ways to use it:
  - Browse: pick a chest X-ray from a folder of examples and read the generated
    report next to the ground-truth reference (when available).
  - Upload: drop in your own chest X-ray and generate a report live.

Both paths run the model live via src.inference.ReportGenerator. Point the
sidebar at a trained checkpoint (best.pt) and a folder of images.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit runs this file as a script, so the repo root isn't on sys.path by
# default and `import src...` would fail. Add it explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
from PIL import Image

from src.inference import ReportGenerator

st.set_page_config(
    page_title="Chest X-Ray Report Generator",
    page_icon="🩻",
    layout="wide",
)

DEFAULT_CHECKPOINT = "outputs/transformer/best.pt"
DEFAULT_IMAGES_DIR = "app/images"
DEFAULT_FINDINGS_JSON = "app/images/findings.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_generator(checkpoint_path: str) -> ReportGenerator:
    return ReportGenerator.from_checkpoint(checkpoint_path)


@st.cache_data(show_spinner=False)
def load_references(findings_json: str) -> dict:
    """
    Build reference-report lookups from a JSON file in the R2Gen annotation
    format. Each entry: {"id", "report", "image_path": [...], "split"}.

    Accepts either a flat list of entries or the full annotation.json shape
    ({"train": [...], "val": [...], "test": [...]}). Returns two maps:
      - by_path:  image_path (e.g. "CXR368_IM-1832/0.png") -> report
      - by_study: study id   (e.g. "CXR368_IM-1832")       -> report
    Empty maps if the file is absent or unreadable.
    """
    empty = {"by_path": {}, "by_study": {}}
    path = Path(findings_json)
    if not findings_json or not path.exists():
        return empty
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return empty

    if isinstance(raw, dict):
        entries = [e for v in raw.values() if isinstance(v, list) for e in v]
    else:
        entries = raw

    by_path, by_study = {}, {}
    for e in entries:
        report = e.get("report", "")
        study_id = e.get("id")
        if study_id is not None:
            by_study[study_id] = report
        for rel in e.get("image_path", []):
            by_path[rel] = report
    return {"by_path": by_path, "by_study": by_study}


@st.cache_data(show_spinner=False)
def list_images(images_dir: str) -> list[str]:
    """Relative paths of all images under images_dir, sorted. Recurses into study folders."""
    root = Path(images_dir)
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(str(p.relative_to(root)) for p in files)


def lookup_reference(refs: dict, image_rel: str) -> str | None:
    """
    Reference report for a displayed image. Tries the full relative path first,
    then the study-id folder name (the report is per-study, shared across views).
    """
    by_path = refs.get("by_path", {})
    if image_rel in by_path:
        return by_path[image_rel]
    study_id = Path(image_rel).parent.name  # "CXR368_IM-1832/0.png" -> "CXR368_IM-1832"
    by_study = refs.get("by_study", {})
    if study_id and study_id in by_study:
        return by_study[study_id]
    return None


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")
checkpoint_path = st.sidebar.text_input("Checkpoint (best.pt)", DEFAULT_CHECKPOINT)
images_dir = st.sidebar.text_input("Images folder", DEFAULT_IMAGES_DIR)
findings_json = st.sidebar.text_input("Findings JSON (optional)", DEFAULT_FINDINGS_JSON)
max_length = st.sidebar.slider("Max report length (tokens)", 20, 100, 60)

st.sidebar.caption(
    "Store images preserving their study folders (e.g. `CXR368_IM-1832/0.png`) so "
    "paths match the findings JSON. The JSON is the R2Gen annotation format "
    "({id, report, image_path, split}); it supplies the ground-truth reference."
)


# -----------------------------------------------------------------------------
# Header + model load
# -----------------------------------------------------------------------------
st.title("Chest X-Ray Report Generator")
st.write(
    "A vanilla Transformer (DenseNet-121 encoder → Transformer decoder) trained "
    "on the IU X-Ray dataset to draft a free-text findings report from a single "
    "chest radiograph."
)

try:
    generator = load_generator(checkpoint_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Set the checkpoint path in the sidebar to a trained `best.pt`. "
        "If you haven't trained yet, run "
        "`python -m src.training.train --config configs/transformer.yaml`."
    )
    st.stop()
except Exception as e:  # noqa: BLE001 — surface any load error to the user, don't crash
    st.error(f"Could not load the model: {type(e).__name__}: {e}")
    st.stop()

params_m = generator.num_params / 1e6
meta_bits = [f"model: {generator.model_name}", f"{params_m:.1f}M params", f"device: {generator.device.type}"]
if generator.trained_epoch is not None:
    meta_bits.append(f"trained to epoch {generator.trained_epoch}")
if generator.best_metric is not None:
    meta_bits.append(f"best val_loss {generator.best_metric:.4f}")
st.caption("  ·  ".join(meta_bits))

references = load_references(findings_json)

st.divider()


def render_result(image: Image.Image, report: str, reference: str | None = None):
    """Two-column layout: the X-ray on the left, the generated report on the right."""
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(image, caption="Input radiograph", use_container_width=True)
    with right:
        st.subheader("Generated report")
        if report.strip():
            st.write(report)
        else:
            st.warning("The model produced an empty report for this image.")
        if reference is not None:
            st.subheader("Reference report")
            st.write(reference if reference.strip() else "_(no reference text)_")


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
browse_tab, upload_tab = st.tabs(["Browse examples", "Upload an image"])

with browse_tab:
    image_files = list_images(images_dir)
    if not image_files:
        st.info(
            f"No images found in `{images_dir}`. Copy a few chest X-ray PNGs there, "
            f"keeping their study folders (e.g. `CXR368_IM-1832/0.png`), or point the "
            f"sidebar at your dataset's `images/` folder to browse the full test set."
        )
    else:
        choice = st.selectbox(
            f"Choose an image ({len(image_files)} available)",
            image_files,
            index=0,
        )
        image_path = Path(images_dir) / choice
        reference = lookup_reference(references, choice)

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not open {image_path}: {e}")
            st.stop()

        with st.spinner("Generating report…"):
            report = generator.predict(pil_image, max_length=max_length)

        render_result(pil_image, report, reference=reference)

        if reference is None and references["by_path"]:
            st.caption("No matching reference in the findings JSON for this image.")

with upload_tab:
    uploaded = st.file_uploader(
        "Upload a chest X-ray",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info("Upload a single-view chest radiograph (PNG or JPEG) to generate a report.")
    else:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the uploaded file: {e}")
            st.stop()

        with st.spinner("Generating report…"):
            report = generator.predict(pil_image, max_length=max_length)

        render_result(pil_image, report, reference=None)