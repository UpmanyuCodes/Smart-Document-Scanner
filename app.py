"""Smart Document Scanner — mobile-first Streamlit interface.

A CamScanner-style, browser-based scanner designed to feel like a phone app.
Capture pages from the camera or upload them, edit each page with scanner
filters and adjustments (apply to one page or all), and export the edited
multi-page document as PDF / ZIP.

UI only — all image-processing lives in the ``scanner`` and ``utils`` packages.

Build status:
  Phase 1  — foundation, upload, validation, preview            (done)
  Camera + multi-page document + mobile UI                      (done)
  Phase 2  — preprocessing + document boundary detection        (done)
  Phase 3  — perspective correction + deskew                    (done)
  Phase 4  — enhancement (brightness/contrast/CLAHE/sharpen)    (done)
  Phase 5  — scanner filters (grayscale / B&W / high contrast)  (done)
  Phase 6+ — OCR + scan-quality score                           (planned)
"""

from __future__ import annotations

import streamlit as st

from scanner import detection, pipeline, preprocessing
from scanner.pipeline import EditSettings
from utils import document, image_io, validation

st.set_page_config(page_title="Smart Document Scanner", page_icon="📄",
                   layout="centered")

PREVIEW_MAX = 700        # px, live edit preview
THUMB_MAX = 320          # px, document gallery thumbnails


MOBILE_CSS = """
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 460px; padding-top: 0.5rem; padding-bottom: 5rem;
}
.app-bar {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
    color: #fff; border-radius: 16px; padding: 16px 18px; margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(79,70,229,0.25);
}
.app-bar h1 {font-size: 1.35rem; margin: 0; color: #fff;}
.app-bar p {margin: 2px 0 0; font-size: 0.8rem; opacity: 0.92;}
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; padding: 0.55rem 0.5rem; min-height: 2.6rem;
}
[data-testid="stSegmentedControl"] {margin-bottom: 0.5rem;}
.block-container p {line-height: 1.4;}
</style>
"""


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("pages", [])
    st.session_state.setdefault("page_counter", 0)
    st.session_state.setdefault("camera_nonce", 0)
    st.session_state.setdefault("upload_nonce", 0)
    st.session_state.setdefault("detections", {})   # page_id -> DetectionResult
    st.session_state.setdefault("edits", {})        # page_id -> EditSettings
    st.session_state.setdefault("ed_loaded_page", None)


def add_page(image, name: str, source: str) -> None:
    st.session_state["page_counter"] += 1
    st.session_state["pages"].append(
        {"id": st.session_state["page_counter"], "image": image,
         "name": name, "source": source}
    )


def remove_page(page_id: int) -> None:
    st.session_state["pages"] = [p for p in st.session_state["pages"]
                                 if p["id"] != page_id]
    st.session_state["detections"].pop(page_id, None)
    st.session_state["edits"].pop(page_id, None)


def move_page(page_id: int, direction: int) -> None:
    pages = st.session_state["pages"]
    index = next((i for i, p in enumerate(pages) if p["id"] == page_id), None)
    if index is None:
        return
    new_index = index + direction
    if 0 <= new_index < len(pages):
        pages[index], pages[new_index] = pages[new_index], pages[index]


def clear_pages() -> None:
    st.session_state["pages"] = []
    st.session_state["detections"] = {}
    st.session_state["edits"] = {}


def get_detection(page):
    cache = st.session_state["detections"]
    if page["id"] not in cache:
        cache[page["id"]] = detection.detect_document_corners(page["image"])
    return cache[page["id"]]


def corners_for(page):
    det = get_detection(page)
    return det.corners if det.success else None


def ensure_settings(page) -> EditSettings:
    """Return stored edit settings for a page, creating defaults on first use.

    Auto-crop defaults ON when a boundary was detected (CamScanner-like)."""
    edits = st.session_state["edits"]
    if page["id"] not in edits:
        det = get_detection(page)
        edits[page["id"]] = EditSettings(crop=det.success)
    return edits[page["id"]]


def render_final(page, preview_max=None):
    """Render a page with its current edit settings."""
    return pipeline.render_page(
        page["image"], ensure_settings(page), corners_for(page), preview_max
    )


def decode_and_validate(data: bytes):
    size_check = validation.validate_file_size(len(data))
    if not size_check.ok:
        return None, size_check.message
    image = image_io.decode_image(data)
    array_check = validation.validate_image_array(image)
    if not array_check.ok:
        return None, array_check.message
    return image, ""


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------
def app_bar() -> None:
    n = len(st.session_state["pages"])
    st.markdown(
        f"""<div class="app-bar"><h1>📄 Smart Document Scanner</h1>
        <p>Capture • Detect • Correct • Enhance • Extract &nbsp;·&nbsp;
        {n} page(s)</p></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Screen: Capture
# --------------------------------------------------------------------------
def screen_capture() -> None:
    st.subheader("📷 Add a page")
    mode = st.segmented_control("Source", ["Camera", "Upload"], default="Camera",
                                key="capture_mode", label_visibility="collapsed") \
        or "Camera"

    with st.container(border=True):
        if mode == "Camera":
            photo = st.camera_input("Point at the document and capture",
                                    key=f"camera_{st.session_state['camera_nonce']}")
            if photo is not None and st.button("➕ Add this page", type="primary",
                                               use_container_width=True):
                image, error = decode_and_validate(photo.getvalue())
                if error:
                    st.error(error)
                else:
                    n = len(st.session_state["pages"]) + 1
                    add_page(image, f"Camera page {n}", "camera")
                    st.session_state["camera_nonce"] += 1
                    st.toast("Page added")
                    st.rerun()
        else:
            uploads = st.file_uploader(
                "Upload image(s)", type=list(validation.SUPPORTED_EXTENSIONS),
                accept_multiple_files=True,
                key=f"upload_{st.session_state['upload_nonce']}")
            if uploads and st.button(f"➕ Add {len(uploads)} page(s)", type="primary",
                                     use_container_width=True):
                added, errors = 0, []
                for up in uploads:
                    ext_check = validation.validate_extension(up.name)
                    if not ext_check.ok:
                        errors.append(f"{up.name}: {ext_check.message}")
                        continue
                    image, error = decode_and_validate(up.getvalue())
                    if error:
                        errors.append(f"{up.name}: {error}")
                        continue
                    add_page(image, up.name, "upload")
                    added += 1
                for message in errors:
                    st.warning(message)
                if added:
                    st.session_state["upload_nonce"] += 1
                    st.toast(f"Added {added} page(s)")
                    st.rerun()

    st.caption("📸 Camera works over localhost without HTTPS. Capture pages one "
               "after another, then open **Edit** to apply scanner filters.")


# --------------------------------------------------------------------------
# Screen: Edit (CamScanner-style, per page or all pages)
# --------------------------------------------------------------------------
_CONTROL_DEFAULTS = {
    "ed_filter": "filter_mode", "ed_crop": "crop", "ed_deskew": "deskew",
    "ed_rotate": "rotate", "ed_bright": "brightness", "ed_contrast": "contrast",
    "ed_sharpen": "sharpen", "ed_block": "bw_block_size", "ed_c": "bw_c",
}


def _seed_controls(settings: EditSettings) -> None:
    """Load a page's settings into the edit widgets' session_state (overwrite)."""
    for key, attr in _CONTROL_DEFAULTS.items():
        st.session_state[key] = getattr(settings, attr)


def _ensure_control_keys(settings: EditSettings) -> None:
    """Seed any control keys that are missing (e.g. a conditionally-shown
    slider whose state Streamlit cleaned up) without overwriting live values."""
    for key, attr in _CONTROL_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = getattr(settings, attr)


def screen_edit() -> None:
    pages = st.session_state["pages"]
    if not pages:
        st.info("No pages yet. Add one from the **Capture** tab.")
        return

    st.subheader("✏️ Edit page")
    labels = {f"Page {i + 1}": p for i, p in enumerate(pages)}
    choice = st.selectbox("Page", list(labels.keys()), label_visibility="collapsed")
    page = labels[choice]
    settings = ensure_settings(page)
    det = get_detection(page)

    # When the selected page changes, load its stored settings into the widgets.
    if st.session_state["ed_loaded_page"] != page["id"]:
        _seed_controls(settings)
        st.session_state["ed_loaded_page"] = page["id"]
    _ensure_control_keys(settings)

    # --- Filter picker ---
    with st.container(border=True):
        st.markdown("**Filter**")
        filter_mode = st.segmented_control(
            "Filter", pipeline.FILTER_MODES, key="ed_filter",
            label_visibility="collapsed") or pipeline.FILTER_ORIGINAL

        # --- Geometry row ---
        col_a, col_b = st.columns(2)
        with col_a:
            crop = st.checkbox("Auto-crop edges", key="ed_crop",
                               disabled=not det.success,
                               help=None if det.success else
                               "No document boundary detected for this page.")
            deskew = st.checkbox("Auto-straighten", key="ed_deskew")
        with col_b:
            if st.button("🔄 Rotate 90°", use_container_width=True):
                st.session_state["ed_rotate"] = (st.session_state["ed_rotate"] + 90) % 360
                st.rerun()
            st.caption(f"Rotation: {st.session_state['ed_rotate']}°")

        # --- Adjustments ---
        brightness = st.slider("Brightness", -100, 100, key="ed_bright")
        contrast = st.slider("Contrast", 0.5, 2.0, step=0.05, key="ed_contrast")
        if filter_mode != pipeline.FILTER_BW:
            sharpen = st.slider("Sharpen", 0.0, 3.0, step=0.1, key="ed_sharpen")
        else:
            sharpen = st.session_state["ed_sharpen"]

        # B&W-specific controls
        if filter_mode == pipeline.FILTER_BW:
            block = st.slider("B&W detail (block size)", 3, 51, step=2, key="ed_block")
            c_val = st.slider("B&W threshold bias", -20, 30, key="ed_c")
        else:
            block = st.session_state["ed_block"]
            c_val = st.session_state["ed_c"]

    # --- Persist the edited settings for this page ---
    new_settings = EditSettings(
        crop=crop, deskew=deskew, rotate=st.session_state["ed_rotate"],
        brightness=brightness, contrast=contrast, sharpen=sharpen,
        filter_mode=filter_mode, bw_block_size=block, bw_c=c_val,
    )
    st.session_state["edits"][page["id"]] = new_settings

    # --- Live preview (before / after) ---
    st.markdown("**Preview**")
    col_before, col_after = st.columns(2)
    with col_before:
        st.image(image_io.to_display(pipeline.render_page(
            page["image"], EditSettings(), None, PREVIEW_MAX)),
            caption="Original", use_container_width=True)
    with col_after:
        preview = pipeline.render_page(page["image"], new_settings,
                                       corners_for(page), PREVIEW_MAX)
        st.image(preview, caption="Edited", use_container_width=True)

    # --- Apply / reset ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Apply to all pages", type="primary",
                     use_container_width=True):
            for p in pages:
                s = new_settings.copy()
                # crop only where that page actually has a detected boundary
                s.crop = new_settings.crop and corners_for(p) is not None
                st.session_state["edits"][p["id"]] = s
            st.toast("Applied to all pages")
    with col2:
        if st.button("↩️ Reset this page", use_container_width=True):
            st.session_state["edits"][page["id"]] = EditSettings(crop=det.success)
            st.session_state["ed_loaded_page"] = None  # force reseed
            st.rerun()

    if not det.success:
        st.caption(f"ℹ️ {det.message}")

    with st.expander("🔬 Processing steps (for the lab demo)"):
        gray = preprocessing.to_grayscale(page["image"])
        edges = preprocessing.canny_edges(page["image"])
        st.image(gray, caption="Grayscale", use_container_width=True)
        st.image(edges, caption="Canny edges", use_container_width=True)
        if det.success:
            overlay = detection.draw_corners(page["image"], det.corners)
            st.image(image_io.to_display(overlay),
                     caption="Detected boundary", use_container_width=True)


# --------------------------------------------------------------------------
# Screen: Document (review + export of edited pages)
# --------------------------------------------------------------------------
def screen_document() -> None:
    pages = st.session_state["pages"]
    if not pages:
        st.info("No pages yet. Add one from the **Capture** tab.")
        return

    st.subheader(f"📑 Document — {len(pages)} page(s)")
    st.caption("Previews show your edits. Reorder or remove pages; this is the "
               "export order.")

    for position, page in enumerate(pages, start=1):
        with st.container(border=True):
            col_img, col_ctl = st.columns([2, 1])
            with col_img:
                st.image(image_io.to_display(render_final(page, THUMB_MAX)),
                         use_container_width=True)
            with col_ctl:
                st.markdown(f"**Page {position}**")
                st.caption(ensure_settings(page).filter_mode)
                if st.button("⬆️ Up", key=f"up_{page['id']}",
                             use_container_width=True, disabled=position == 1):
                    move_page(page["id"], -1)
                    st.rerun()
                if st.button("⬇️ Down", key=f"down_{page['id']}",
                             use_container_width=True,
                             disabled=position == len(pages)):
                    move_page(page["id"], +1)
                    st.rerun()
                if st.button("🗑️ Remove", key=f"del_{page['id']}",
                             use_container_width=True):
                    remove_page(page["id"])
                    st.rerun()

    st.divider()
    _export_block(pages)

    if st.button("🗑️ Clear all pages", use_container_width=True):
        clear_pages()
        st.rerun()


def _export_block(pages) -> None:
    st.markdown("### ⬇️ Export edited document")
    with st.spinner("Rendering pages at full resolution…"):
        images = [render_final(p) for p in pages]  # full-res edited pages

    doc_name = st.text_input("Document name", value="scanned_document")
    safe = (doc_name.strip() or "scanned_document").replace(" ", "_")

    col_pdf, col_zip = st.columns(2)
    with col_pdf:
        pdf = document.images_to_pdf_bytes(images)
        st.download_button("📄 PDF", data=pdf or b"", file_name=f"{safe}.pdf",
                           mime="application/pdf", disabled=pdf is None,
                           use_container_width=True)
    with col_zip:
        zipped = document.images_to_zip_bytes(images, ".png")
        st.download_button("🗂️ Images (ZIP)", data=zipped or b"",
                           file_name=f"{safe}_images.zip", mime="application/zip",
                           disabled=zipped is None, use_container_width=True)
    st.caption("PDF/ZIP contain each page with its applied crop, filter, and "
               "adjustments, in the order above.")


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    _init_state()
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    app_bar()

    nav = st.segmented_control(
        "Navigation", ["📷 Capture", "✏️ Edit", "📑 Document"],
        default="📷 Capture", key="nav", label_visibility="collapsed") \
        or "📷 Capture"

    if nav == "📷 Capture":
        screen_capture()
    elif nav == "✏️ Edit":
        screen_edit()
    else:
        screen_document()


if __name__ == "__main__":
    main()
