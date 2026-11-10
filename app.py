"""Smart Document Scanner — mobile-first Streamlit interface.

A CamScanner-style, browser-based scanner designed to feel like a phone app:
a narrow centered frame, a top app bar, segmented navigation, card sections,
and large touch-friendly buttons. Capture pages from the camera or upload
them, build a multi-page document, auto-detect the document boundary, and
export as PDF / ZIP.

UI only — all image-processing and document-assembly logic lives in the
``scanner`` and ``utils`` packages.

Build status:
  Phase 1  — foundation, upload, validation, preview            (done)
  Camera + multi-page document capture/assembly                 (done)
  Phase 2  — preprocessing + document boundary detection        (done)
  Phase 3+ — perspective, enhancement, filters, OCR, quality    (planned)
"""

from __future__ import annotations

import streamlit as st

from scanner import detection, preprocessing
from utils import document, image_io, validation

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="centered",
)


# --------------------------------------------------------------------------
# Mobile-app styling
# --------------------------------------------------------------------------
MOBILE_CSS = """
<style>
/* Hide Streamlit chrome for an app-like surface */
#MainMenu, header, footer {visibility: hidden;}

/* Phone-width centered frame */
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 460px;
    padding-top: 0.5rem;
    padding-bottom: 5rem;
}

/* App bar */
.app-bar {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
    color: #fff;
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(79,70,229,0.25);
}
.app-bar h1 {font-size: 1.35rem; margin: 0; color: #fff;}
.app-bar p {margin: 2px 0 0; font-size: 0.8rem; opacity: 0.92;}

/* Buttons: full width, rounded, touch-friendly */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55rem 0.5rem;
    min-height: 2.6rem;
}

/* Segmented nav spacing */
[data-testid="stSegmentedControl"] {margin-bottom: 0.5rem;}

/* Tighter captions */
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
    st.session_state.setdefault("detections", {})  # page_id -> DetectionResult


def add_page(image, name: str, source: str) -> None:
    st.session_state["page_counter"] += 1
    st.session_state["pages"].append(
        {
            "id": st.session_state["page_counter"],
            "image": image,
            "name": name,
            "source": source,
        }
    )


def remove_page(page_id: int) -> None:
    st.session_state["pages"] = [
        p for p in st.session_state["pages"] if p["id"] != page_id
    ]
    st.session_state["detections"].pop(page_id, None)


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


def get_detection(page):
    """Run (and cache) document detection for a page."""
    cache = st.session_state["detections"]
    if page["id"] not in cache:
        cache[page["id"]] = detection.detect_document_corners(page["image"])
    return cache[page["id"]]


# --------------------------------------------------------------------------
# Decode + validate (shared by camera and upload)
# --------------------------------------------------------------------------
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
# Screens
# --------------------------------------------------------------------------
def app_bar() -> None:
    page_count = len(st.session_state["pages"])
    st.markdown(
        f"""
        <div class="app-bar">
            <h1>📄 Smart Document Scanner</h1>
            <p>Capture • Detect • Correct • Enhance • Extract &nbsp;·&nbsp;
               {page_count} page(s)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def screen_capture() -> None:
    st.subheader("📷 Add a page")
    mode = st.segmented_control(
        "Source", ["Camera", "Upload"], default="Camera",
        key="capture_mode", label_visibility="collapsed",
    ) or "Camera"

    with st.container(border=True):
        if mode == "Camera":
            photo = st.camera_input(
                "Point at the document and capture",
                key=f"camera_{st.session_state['camera_nonce']}",
            )
            if photo is not None:
                if st.button("➕ Add this page", type="primary",
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
                "Upload image(s)",
                type=list(validation.SUPPORTED_EXTENSIONS),
                accept_multiple_files=True,
                key=f"upload_{st.session_state['upload_nonce']}",
            )
            if uploads and st.button(
                f"➕ Add {len(uploads)} page(s)", type="primary",
                use_container_width=True,
            ):
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

    st.caption(
        "📸 Camera works over localhost without HTTPS; your browser will ask "
        "for permission. Capture pages one after another — review and export "
        "them from the **Document** tab."
    )


def screen_document() -> None:
    pages = st.session_state["pages"]
    if not pages:
        st.info("No pages yet. Add one from the **Capture** tab.")
        return

    st.subheader(f"📑 Document — {len(pages)} page(s)")
    st.caption("Reorder or remove pages; this is the export order.")

    for position, page in enumerate(pages, start=1):
        with st.container(border=True):
            col_img, col_ctl = st.columns([2, 1])
            with col_img:
                st.image(image_io.to_display(page["image"]),
                         use_container_width=True)
            with col_ctl:
                st.markdown(f"**Page {position}**")
                st.caption(page["source"])
                if st.button("⬆️ Up", key=f"up_{page['id']}",
                             use_container_width=True, disabled=position == 1):
                    move_page(page["id"], -1)
                    st.rerun()
                if st.button("⬇️ Down", key=f"down_{page['id']}",
                             use_container_width=True, disabled=position == len(pages)):
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
    st.markdown("### ⬇️ Export")
    images = [p["image"] for p in pages]
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
    st.caption(
        "Currently exports the captured pages. As later phases land, each page "
        "will be perspective-corrected and enhanced before export."
    )


def screen_review() -> None:
    pages = st.session_state["pages"]
    if not pages:
        st.info("No pages yet. Add one from the **Capture** tab.")
        return

    st.subheader("🔍 Review page")
    labels = {f"Page {i + 1}": p for i, p in enumerate(pages)}
    choice = st.selectbox("Page", list(labels.keys()), label_visibility="collapsed")
    page = labels[choice]
    image = page["image"]

    tab_detect, tab_steps, tab_more = st.tabs(
        ["Detection", "Processing steps", "Next phases"]
    )

    with tab_detect:
        result = get_detection(page)
        if result.success:
            overlay = detection.draw_corners(image, result.corners)
            st.image(image_io.to_display(overlay), use_container_width=True,
                     caption="Detected document boundary")
            st.success(
                f"✓ Document detected — covers {result.area_ratio * 100:.0f}% "
                "of the frame."
            )
            with st.expander("Corner coordinates (original image)"):
                names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
                for name, (x, y) in zip(names, result.corners):
                    st.write(f"- {name}: ({x:.0f}, {y:.0f})")
        else:
            st.image(image_io.to_display(image), use_container_width=True)
            st.warning(f"⚠️ {result.message}")
            st.caption(
                "Detection is heuristic. Try a plainer background with the whole "
                "page visible and good contrast between paper and surface. "
                "Perspective correction with manual corners is planned for Phase 3."
            )

    with tab_steps:
        st.caption("Intermediate stages used for detection — useful for the "
                   "lab demonstration.")
        gray = preprocessing.to_grayscale(image)
        blurred = preprocessing.gaussian_blur(gray, 5)
        edges = preprocessing.canny_edges(image)
        st.image(gray, caption="1 · Grayscale", use_container_width=True)
        st.image(blurred, caption="2 · Gaussian blur", use_container_width=True)
        st.image(edges, caption="3 · Canny edges", use_container_width=True)

    with tab_more:
        st.info(
            "🚧 Coming next: perspective correction, deskew, enhancement & "
            "filters, OCR text extraction, and the scan-quality score."
        )


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
def main() -> None:
    _init_state()
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    app_bar()

    nav = st.segmented_control(
        "Navigation",
        ["📷 Capture", "📑 Document", "🔍 Review"],
        default="📷 Capture",
        key="nav",
        label_visibility="collapsed",
    ) or "📷 Capture"

    if nav == "📷 Capture":
        screen_capture()
    elif nav == "📑 Document":
        screen_document()
    else:
        screen_review()


if __name__ == "__main__":
    main()
