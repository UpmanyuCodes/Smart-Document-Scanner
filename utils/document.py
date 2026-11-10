"""Multi-page document assembly.

Combines one or more processed page images into a single downloadable
document. PDF assembly uses Pillow (already a dependency) so no extra
packages are required. All inputs are OpenCV BGR (or grayscale) arrays and
are converted to RGB before being written.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from utils import image_io


def _to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR/grayscale OpenCV array to an RGB Pillow image."""
    rgb = image_io.bgr_to_rgb(image)
    if rgb.ndim == 2:  # grayscale -> promote to RGB for consistent PDF pages
        return Image.fromarray(rgb).convert("RGB")
    return Image.fromarray(rgb).convert("RGB")


def images_to_pdf_bytes(images: list[np.ndarray]) -> bytes | None:
    """Assemble a list of images into a single multi-page PDF.

    Page order follows list order. Returns ``None`` if the list is empty or
    assembly fails.
    """
    if not images:
        return None
    try:
        pages = [_to_pil(img) for img in images if img is not None]
        if not pages:
            return None
        buffer = io.BytesIO()
        pages[0].save(
            buffer,
            format="PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=150.0,
        )
        return buffer.getvalue()
    except Exception:
        return None


def images_to_zip_bytes(images: list[np.ndarray], ext: str = ".png") -> bytes | None:
    """Bundle page images into a ZIP archive (one image file per page).

    Useful when the user wants the individual processed images rather than a
    combined PDF. Returns ``None`` on failure or empty input.
    """
    if not images:
        return None
    import zipfile

    try:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, img in enumerate(images, start=1):
                if img is None:
                    continue
                data = image_io.encode_image(img, ext)
                if data is None:
                    continue
                archive.writestr(f"page_{index:02d}{ext}", data)
        return buffer.getvalue()
    except Exception:
        return None
