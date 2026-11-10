"""Headless smoke tests for the Streamlit UI (via streamlit.testing.AppTest).

They render every screen, with and without pages, and drive key interactions
to make sure no screen raises and edits reach the stored page settings.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from scanner import pipeline

# Absolute path: newer Streamlit resolves relative paths against this file.
APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _document_photo(shade=235):
    img = np.full((600, 800, 3), 40, dtype=np.uint8)
    cv2.rectangle(img, (150, 80), (650, 520), (shade, shade, shade), -1)
    cv2.putText(img, "INVOICE 42", (190, 300), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (20, 20, 20), 5)
    return img


def _app(with_pages=0, nav="scan"):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    if with_pages:
        at.session_state["pages"] = [
            {"id": i + 1, "image": _document_photo(230 - i * 10), "source": "upload"}
            for i in range(with_pages)
        ]
        at.session_state["page_counter"] = with_pages
    at.session_state["nav"] = nav
    return at.run()


@pytest.mark.parametrize("nav", ["scan", "edit", "export"])
def test_screens_render_without_pages(nav):
    at = _app(0, nav)
    assert not at.exception


@pytest.mark.parametrize("nav", ["scan", "edit", "export"])
def test_screens_render_with_pages(nav):
    at = _app(3, nav)
    assert not at.exception


def test_filter_applies_to_current_page_only():
    at = _app(2, "edit")
    at.button(key="fbtn_3").click().run()  # B&W
    assert not at.exception
    edits = at.session_state["edits"]
    assert edits[1].filter_mode == pipeline.FILTER_BW
    assert edits[2].filter_mode == pipeline.FILTER_ORIGINAL


def test_filter_applies_to_all_pages_in_all_scope():
    at = _app(3, "edit")
    at.session_state["scope"] = "all"
    at.session_state["w_scope"] = "all"
    at.run()
    at.button(key="fbtn_2").click().run()  # Grayscale
    assert not at.exception
    assert all(s.filter_mode == pipeline.FILTER_GRAYSCALE
               for s in at.session_state["edits"].values())


def test_auto_crop_defaults_on_when_edges_found():
    at = _app(1, "edit")
    assert at.session_state["edits"][1].crop is True


def test_page_navigation_and_delete():
    at = _app(3, "edit")
    at.button(key="next_page").click().run()
    assert at.session_state["cur"] == 1
    at.session_state["nav"] = "export"
    at.run()
    at.button(key="cd_2").click().run()  # delete page id 2 from its export card
    assert not at.exception
    assert [p["id"] for p in at.session_state["pages"]] == [1, 3]
    assert at.session_state["cur"] == 0


def test_crop_tools_straighten_rotate_and_trim():
    at = _app(1, "edit")
    at.toggle(key="w_deskew_1_0").set_value(True).run()
    assert not at.exception
    assert at.session_state["edits"][1].deskew is True
    at.button(key="rotate_left").click().run()
    assert at.session_state["edits"][1].rotate == 90
    at.button(key="rotate_right").click().run()
    at.button(key="rotate_right").click().run()
    assert at.session_state["edits"][1].rotate == 270
    at.slider(key="w_margin_top_1_0").set_value(10).run()
    assert not at.exception
    assert at.session_state["edits"][1].margin_top == 10


def test_adjustments_and_quality_tab_render():
    at = _app(1, "edit")
    at.slider(key="w_brightness_1_0").set_value(30).run()
    at.toggle(key="w_clean_background_1_0").set_value(True).run()
    assert not at.exception
    settings = at.session_state["edits"][1]
    assert settings.brightness == 30 and settings.clean_background is True


def test_new_document_dialog_clears_pages():
    at = _app(2, "export")
    next(b for b in at.button if b.label == "Start a new document").click().run()
    remove = [b for b in at.button if b.label == "Remove pages"]
    assert remove, "confirmation dialog should offer 'Remove pages'"
    remove[0].click().run()
    assert not at.exception
    assert at.session_state["pages"] == []
    assert at.session_state["nav"] == "scan"


def test_export_formats_render():
    at = _app(2, "export")
    for fmt in ("PDF", "Images"):
        at.session_state["export_format"] = fmt
        at.session_state["w_format"] = fmt
        at.run()
        assert not at.exception
        assert any(b.label.startswith("Download") for b in at.get("download_button"))


def test_settings_from_older_app_version_are_upgraded():
    """Reproduces a session that outlived a code update: stored settings are
    instances of an old class without key()/geometry_key()."""

    class OldEditSettings:  # shape of the settings before the redesign
        def __init__(self):
            self.crop, self.deskew, self.rotate = True, False, 90
            self.brightness, self.contrast, self.sharpen = 15, 1.2, 0.0
            self.filter_mode, self.bw_block_size, self.bw_c = "Grayscale", 21, 10

    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["pages"] = [{"id": 1, "image": _document_photo(), "source": "upload"}]
    at.session_state["page_counter"] = 1
    at.session_state["edits"] = {1: OldEditSettings()}
    for nav in ("scan", "edit", "export"):
        at.session_state["nav"] = nav
        at.run()
        assert not at.exception, nav
    upgraded = at.session_state["edits"][1]
    assert hasattr(upgraded, "key")
    assert (upgraded.rotate, upgraded.brightness, upgraded.filter_mode) == (90, 15, "Grayscale")
    assert upgraded.denoise == 0  # new field gets its default


def test_original_images_untouched_by_editing():
    at = _app(1, "edit")
    before = at.session_state["pages"][0]["image"].copy()
    at.button(key="fbtn_1").click().run()  # Magic
    assert np.array_equal(at.session_state["pages"][0]["image"], before)
