"""Smart Document Scanner — Streamlit interface.

This is the user-facing layer only. All image-processing logic lives in the
``scanner`` and ``utils`` packages so the UI stays thin and the processing
stays testable.

Build status: Phase 1 (project foundation + image upload & preview).
Later phases add detection, perspective correction, enhancement, filters,
OCR, and the quality analyzer.
"""

from __future__ import annotations

import streamlit as st

from utils import image_io, validation

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide",
)


def render_header() -> None:
    st.title("📄 Smart Document Scanner")
    st.caption("**Detect • Correct • Enhance • Analyze • Extract**")
    st.write(
        "Turn a photograph of a physical document into a corrected, enhanced, "
        "readable scan. Upload a photo to begin. Your image is processed locally "
        "and is never uploaded to any external service."
    )


def load_uploaded_image(uploaded_file):
    """Validate and decode an uploaded file.

    Returns the decoded BGR image on success, or ``None`` after showing an
    error. Validation happens before and after decoding so corrupt files,
    unsupported types, and degenerate sizes are all caught gracefully.
    """
    data = uploaded_file.getvalue()

    ext_check = validation.validate_extension(uploaded_file.name)
    if not ext_check.ok:
        st.error(ext_check.message)
        return None

    size_check = validation.validate_file_size(len(data))
    if not size_check.ok:
        st.error(size_check.message)
        return None

    image = image_io.decode_image(data)
    array_check = validation.validate_image_array(image)
    if not array_check.ok:
        st.error(array_check.message)
        return None

    if array_check.warning:
        st.warning(array_check.warning)

    return image


def main() -> None:
    render_header()

    # --- Sidebar: upload and (future) controls ---
    with st.sidebar:
        st.header("1 · Upload")
        uploaded_file = st.file_uploader(
            "Upload a document photo",
            type=list(validation.SUPPORTED_EXTENSIONS),
            help="JPG, JPEG, or PNG. The original is preserved; all processing "
            "runs on copies.",
        )
        st.divider()
        st.caption("More controls (Auto Scan, enhancement sliders, OCR) "
                   "appear here as later build phases land.")

    if uploaded_file is None:
        st.info("⬅️ Upload a JPG, JPEG, or PNG photo of a document to get started.")
        return

    original = load_uploaded_image(uploaded_file)
    if original is None:
        return

    # Keep an untouched copy of the original in session state. All processing
    # in later phases operates on copies, never this array.
    st.session_state["original_image"] = original

    width, height, channels = image_io.image_dimensions(original)

    # --- Main area: tabbed layout (sections fill in across phases) ---
    tab_original, tab_detection, tab_corrected, tab_enhance, tab_ocr, tab_quality = st.tabs(
        [
            "Original Image",
            "Document Detection",
            "Corrected Scan",
            "Enhancement & Filters",
            "OCR Text",
            "Quality Report",
        ]
    )

    with tab_original:
        st.subheader("Original Image")
        col_img, col_info = st.columns([3, 1])
        with col_img:
            st.image(
                image_io.to_display(original),
                caption=uploaded_file.name,
                use_container_width=True,
            )
        with col_info:
            st.markdown("**Image details**")
            st.write(f"- Width: **{width}px**")
            st.write(f"- Height: **{height}px**")
            st.write(f"- Channels: **{channels}**")
            st.write(f"- Size: **{len(uploaded_file.getvalue()) / 1024:.0f} KB**")

    for tab, name in [
        (tab_detection, "Document Detection"),
        (tab_corrected, "Corrected Scan"),
        (tab_enhance, "Enhancement & Filters"),
        (tab_ocr, "OCR Text"),
        (tab_quality, "Quality Report"),
    ]:
        with tab:
            st.subheader(name)
            st.info(f"🚧 {name} is coming in a later build phase.")


if __name__ == "__main__":
    main()
