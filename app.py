"""Smart Document Scanner — Streamlit app.

Mobile-first document scanner: capture or upload pages, edit them with
scanner filters and adjustments (per page or across all pages), check scan
quality, extract text, and export the document as PDF, images, or text.

This module is presentation only; all image processing lives in ``scanner``
and ``utils``.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import streamlit as st

from scanner import detection, ocr, pipeline, preprocessing, quality
from scanner.pipeline import EditSettings
from utils import document, image_io, validation

st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ss = st.session_state

# Render sizes (longest side, px). Previews are downscaled for speed; exports
# are always rendered at full resolution.
PREVIEW_SIZE = 900
CARD_SIZE = 600
TILE_SIZE = 240
THUMB_SIZE = 180
CACHE_LIMIT = 48  # per cache, per session; keeps memory modest on shared hosts

SCOPE_PAGE = "page"
SCOPE_ALL = "all"

NAV_LABELS = {
    "scan": ":material/document_scanner: Scan",
    "edit": ":material/tune: Edit",
    "export": ":material/ios_share: Export",
}

FILTER_LABELS = {
    pipeline.FILTER_ORIGINAL: "Original",
    pipeline.FILTER_MAGIC: "Magic",
    pipeline.FILTER_GRAYSCALE: "Grayscale",
    pipeline.FILTER_BW: "B&W",
    pipeline.FILTER_HIGH_CONTRAST: "Contrast",
}

# CSS variables (defined in APP_CSS) so the colours adapt to light/dark mode.
QUALITY_COLORS = {
    "Excellent": "var(--green)",
    "Good": "var(--brand)",
    "Fair": "var(--orange)",
    "Poor": "var(--red)",
}

LOGO_SVG = """<svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32"
rx="8" fill="#007AFF"/><path d="M11 7.5h7l5.5 5.5v11.5a1.5 1.5 0 0 1-1.5 1.5H11a1.5
1.5 0 0 1-1.5-1.5V9a1.5 1.5 0 0 1 1.5-1.5z" fill="#fff"/><path d="M18 7.5V13h5.5"
fill="#B3D7FF"/><rect x="12.5" y="16" width="8" height="1.7" rx=".85" fill="#007AFF"/>
<rect x="12.5" y="19.6" width="5.5" height="1.7" rx=".85" fill="#80BDFF"/></svg>"""

# Colours come from the .empty .e-* rules in APP_CSS (light/dark aware).
EMPTY_SVG = """<svg viewBox="0 0 96 96" aria-hidden="true"><rect class="e-bg" x="4" y="4"
width="88" height="88" rx="28"/><path class="e-page" d="M34 24h20l12 12v34a3 3 0 0 1-3 3H34a3
3 0 0 1-3-3V27a3 3 0 0 1 3-3z" stroke-width="2"/><path class="e-fold" d="M54 24v12h12"/>
<rect class="e-l1" x="37" y="45" width="20" height="3" rx="1.5"/><rect class="e-l2" x="37"
y="52" width="14" height="3" rx="1.5"/><path class="e-frame" d="M22 34v-8a4 4 0 0 1 4-4h8M74
34v-8a4 4 0 0 0-4-4h-8M22 62v8a4 4 0 0 0 4 4h8M74 62v8a4 4 0 0 1-4 4h-8" fill="none"
stroke-width="3" stroke-linecap="round"/></svg>"""

APP_CSS = """
<style>
/* ---- Design tokens (iOS system palette); dark values follow the device ---- */
:root {
    --brand:#007AFF; --brand-soft:rgba(0,122,255,0.12); --sel-ink:#007AFF;
    --bg:#F2F2F7; --surface:#FFFFFF; --ink:#000000; --muted:#6C6C70;
    --line:rgba(60,60,67,0.16); --fill:rgba(118,118,128,0.12);
    --fill-press:rgba(118,118,128,0.2); --thumb:#FFFFFF; --stage:#E5E5EA;
    --bar:rgba(242,242,247,0.8); --glass:rgba(255,255,255,0.74);
    --tint-a:#E5F1FF; --tint-b:#99CAFF; --tint-c:#CCE4FF;
    --green:#34C759; --orange:#FF9500; --red:#FF3B30;
    --shadow:0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.06);
    --thumb-shadow:0 3px 8px rgba(0,0,0,0.12), 0 1px 1px rgba(0,0,0,0.04);
    --ease:cubic-bezier(0.32,0.72,0,1);
}
@media (prefers-color-scheme: dark) {
    :root {
        --brand:#0A84FF; --brand-soft:rgba(10,132,255,0.2); --sel-ink:#FFFFFF;
        --bg:#000000; --surface:#1C1C1E; --ink:#FFFFFF; --muted:#98989F;
        --line:rgba(84,84,88,0.6); --fill:rgba(118,118,128,0.24);
        --fill-press:rgba(118,118,128,0.36); --thumb:#636366; --stage:#1C1C1E;
        --bar:rgba(0,0,0,0.72); --glass:rgba(30,30,32,0.74);
        --tint-a:#0B2545; --tint-b:#2F6FB8; --tint-c:#173A66;
        --green:#30D158; --orange:#FF9F0A; --red:#FF453A;
        --shadow:0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.4);
        --thumb-shadow:0 3px 8px rgba(0,0,0,0.3), 0 1px 1px rgba(0,0,0,0.2);
    }
}

header[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

.stApp { -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
         -webkit-tap-highlight-color:transparent; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 480px; padding: 0.9rem 1rem calc(8.5rem + var(--nav-lift, 0px));
}
/* Keep content steady while the app recomputes instead of dimming it */
[data-stale="true"] { opacity:1 !important; transition:none !important; }

/* ---- Top bar: frosted, stays put while content scrolls beneath ---- */
[data-testid="stElementContainer"]:has(.topbar) {
    position: sticky; top: 0; z-index: 60;
    margin: -0.9rem 0 -6px; padding: 10px 0 8px; background: var(--bar);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    backdrop-filter: saturate(180%) blur(20px);
}
[data-testid="stMarkdownContainer"]:has(> .topbar) { margin-bottom: 0 !important; }
.topbar { display:flex; align-items:center; justify-content:space-between; }
.brand { display:flex; align-items:center; gap:9px; font-weight:600;
         font-size:0.98rem; color:var(--ink); letter-spacing:-0.015em; }
.brand svg { width:28px; height:28px; flex-shrink:0; }
.pill { background:var(--brand-soft); color:var(--brand); font-weight:600;
        font-size:0.76rem; padding:4px 11px; border-radius:999px; white-space:nowrap; }

/* ---- Typography (!important: Streamlit styles markdown paragraphs) ---- */
[data-testid="stMarkdownContainer"]:has(> .muted),
[data-testid="stMarkdownContainer"]:has(> .q-head) { margin-bottom: 0 !important; }
.screen-head { margin: 6px 0 16px; animation: rise 0.45s var(--ease) both; }
.screen-title { font-size:2.05rem !important; font-weight:700; color:var(--ink);
                letter-spacing:-0.025em; line-height:1.15; margin:0 0 4px !important; }
.muted { color:var(--muted) !important; font-size:0.94rem !important;
         line-height:1.4 !important; margin:0 !important; letter-spacing:-0.01em; }
.small { font-size:0.8rem !important; }
.section-label { font-size:0.8rem !important; font-weight:400; letter-spacing:0.02em;
                 text-transform:uppercase; color:var(--muted) !important;
                 margin:8px 0 0 4px !important; }
@keyframes rise { from { opacity:0; transform:translateY(8px); }
                  to { opacity:1; transform:none; } }

/* ---- Buttons ---- */
[data-testid^="stBaseButton-"] {
    transition: transform 0.2s var(--ease), background-color 0.2s, opacity 0.2s;
}
[data-testid^="stBaseButton-"]:active:not(:disabled) { transform: scale(0.97); }
[data-testid^="stBaseButton-"]:focus-visible {
    outline: 3px solid var(--brand-soft); outline-offset: 2px;
}
[data-testid="stBaseButton-primary"] {
    min-height: 50px; border: none !important; border-radius: 14px;
    background: var(--brand) !important; color: #fff !important;
}
[data-testid="stBaseButton-primary"] p { font-size:1.06rem; font-weight:600;
                                         letter-spacing:-0.01em; }
[data-testid="stBaseButton-primary"]:active { opacity: 0.85; }
[data-testid="stBaseButton-secondary"], [data-testid="stPopoverButton"] {
    min-height: 44px; border: none !important; border-radius: 12px;
    background: var(--fill) !important; color: var(--brand) !important;
}
[data-testid="stBaseButton-secondary"] p { font-weight: 500; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover,
[data-testid="stBaseButton-secondary"]:active { background: var(--fill-press) !important; }
[data-testid="stBaseButton-tertiary"] { color: var(--brand) !important; }
[data-testid^="stBaseButton-"]:disabled { opacity: 0.35; }
[class*="st-key-cd_"] button { color: var(--red) !important; }

/* ---- Segmented controls: grey track, raised thumb ---- */
[data-testid="stButtonGroup"] [role="radiogroup"] {
    background: var(--fill); border-radius: 10px; padding: 2px; gap: 0;
    flex-wrap: nowrap;
}
[data-testid^="stBaseButton-segmented_control"] {
    flex: 1 1 0; min-width: 0; min-height: 32px; border: none !important;
    border-radius: 8px !important; background: transparent !important;
    color: var(--ink) !important; margin: 0 !important;
}
[data-testid^="stBaseButton-segmented_control"] p { font-size:0.86rem; font-weight:500; }
[data-testid="stBaseButton-segmented_controlActive"] {
    background: var(--thumb) !important; box-shadow: var(--thumb-shadow);
}
[data-testid="stBaseButton-segmented_controlActive"] p { font-weight: 600; }

/* ---- Floating glass tab bar ---- */
.st-key-bottomnav {
    position: fixed; left: 50%;
    bottom: calc(12px + var(--nav-lift, 0px) + env(safe-area-inset-bottom));
    transform: translateX(-50%); width: min(448px, calc(100% - 24px)); z-index: 999;
    box-sizing: border-box; -webkit-text-size-adjust: 100%; text-size-adjust: 100%;
    background: var(--glass); border: 0.5px solid var(--line); border-radius: 34px;
    -webkit-backdrop-filter: saturate(180%) blur(24px);
    backdrop-filter: saturate(180%) blur(24px);
    padding: 5px; box-shadow: 0 12px 40px rgba(0,0,0,0.16), 0 2px 6px rgba(0,0,0,0.06);
}
.st-key-bottomnav [role="radiogroup"] {
    background: transparent; padding: 0; gap: 4px; width: 100%;
    display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr));
}
.st-key-bottomnav [data-testid^="stBaseButton-segmented_control"] {
    width: 100%; height: 54px; min-height: 54px; border-radius: 29px !important;
    color: var(--muted) !important; box-shadow: none; padding: 4px 0;
    overflow: visible;
}
/* Fixed px sizes + no clipping so large phone font settings can't squash it */
.st-key-bottomnav button > div {
    overflow: visible !important; height: auto !important; max-height: none !important;
}
.st-key-bottomnav button > div > span {
    display: flex; flex-direction: column; align-items: center; gap: 1px;
    flex-shrink: 0;
}
.st-key-bottomnav [data-testid="stIconMaterial"] { font-size: 25px; width: 25px;
                                                   height: 25px; flex-shrink: 0; }
.st-key-bottomnav button > div > span > span { width: 25px; height: 25px;
                                               flex-shrink: 0; }
.st-key-bottomnav button p { font-size: 11px; font-weight: 600; line-height: 14px;
                             white-space: nowrap; }
.st-key-bottomnav [data-testid="stBaseButton-segmented_controlActive"] {
    background: var(--brand-soft) !important; color: var(--brand) !important;
}

/* ---- Tool tabs: icon over label, raised selection ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--fill); border-radius: 14px; padding: 3px; gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    flex: 1 1 0; min-width: 0; height: auto; padding: 7px 0 6px; margin: 0;
    border-radius: 11px; justify-content: center; color: var(--muted);
    transition: background-color 0.25s var(--ease), color 0.2s;
}
[data-testid="stTabs"] [data-baseweb="tab"] p {
    display: flex; flex-direction: column; align-items: center; gap: 2px;
    font-size: 0.68rem; font-weight: 600; line-height: 1.2;
}
[data-testid="stTabs"] [data-baseweb="tab"] p span[role="img"] { font-size: 22px; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    background: var(--thumb); color: var(--sel-ink); box-shadow: var(--thumb-shadow);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 14px; }

/* ---- Grouped rows: sliders and switches sit on their own surface ---- */
[data-testid="stSlider"] {
    background: var(--surface); border-radius: 14px; padding: 10px 16px 4px;
}
[data-testid="stSlider"] label { margin-bottom: 8px; }
[data-testid="stElementContainer"]:has(> [data-testid="stCheckbox"]) {
    width: 100% !important;
}
[data-testid="stSlider"] [role="slider"] {
    width: 26px; height: 26px; background: #fff !important;
    border: 0.5px solid rgba(0,0,0,0.04);
    box-shadow: 0 0.5px 4px rgba(0,0,0,0.12), 0 6px 13px rgba(0,0,0,0.12);
}
[data-testid="stCheckbox"] {
    background: var(--surface); border-radius: 14px; padding: 9px 16px;
    width: 100% !important;
}
[data-testid="stCheckbox"] label {
    flex-direction: row-reverse; justify-content: space-between; align-items: center;
    width: 100%; min-height: 31px; padding: 0; gap: 12px;
}
[data-testid="stCheckbox"] label > div:first-child {
    width: 51px; height: 31px; flex-shrink: 0; margin: 0; padding: 0;
    background: var(--fill-press); transition: background-color 0.25s var(--ease);
}
[data-testid="stCheckbox"] label > div:first-child > div {
    width: 27px; height: 27px; margin: 2px; background: #fff;
    transform: none; transition: transform 0.25s var(--ease);
    box-shadow: 0 3px 8px rgba(0,0,0,0.15), 0 1px 1px rgba(0,0,0,0.06);
}
[data-testid="stCheckbox"] label:has(input:checked) > div:first-child {
    background: var(--green);
}
[data-testid="stCheckbox"] label:has(input:checked) > div:first-child > div {
    transform: translateX(20px);
}
[data-testid="stCheckbox"] label:has(input:disabled) { opacity: 0.45; }

/* ---- Surfaces: cards, fields, sheets ---- */
[class*="st-key-card_"], .st-key-export_panel {
    background: var(--surface); border: none !important; border-radius: 20px;
    box-shadow: var(--shadow);
}
[data-testid="stExpander"] details {
    background: var(--surface); border: none; border-radius: 14px;
}
[data-testid="stExpander"] summary { border-radius: 14px; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {
    background: var(--fill) !important; border-color: transparent !important;
    border-radius: 12px !important;
}
[data-baseweb="base-input"] { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface); border: 1.5px dashed var(--line); border-radius: 20px;
    padding: 22px 16px;
}
[data-testid="stCameraInput"] { border-radius: 20px; overflow: hidden;
                                background: var(--surface); box-shadow: var(--shadow); }
[data-testid="stCameraInput"] button {
    min-height: 50px; border: none !important; background: var(--surface) !important;
    color: var(--brand) !important; font-weight: 600;
}
/* Round shutter button (live preview only; "Clear photo" keeps its text) */
[data-testid="stCameraInput"]:not(:has(img)) [data-testid="stCameraInputButton"] {
    width: 68px !important; height: 68px !important;
    min-width: 68px; min-height: 68px; flex: 0 0 68px;
    display: block; align-self: center; margin: 16px auto 18px !important;
    padding: 0 !important; border-radius: 50% !important;
    font-size: 0 !important; color: transparent !important; overflow: hidden;
    background: var(--brand) !important;
    border: 4px solid var(--surface) !important;
    box-shadow: 0 0 0 3px var(--brand);
    transition: transform .12s ease, box-shadow .12s ease, opacity .12s ease;
    -webkit-tap-highlight-color: transparent;
}
[data-testid="stCameraInput"]:not(:has(img)) [data-testid="stCameraInputButton"]:hover {
    box-shadow: 0 0 0 4px var(--brand);
}
[data-testid="stCameraInput"]:not(:has(img)) [data-testid="stCameraInputButton"]:active {
    transform: scale(.88);
}
[data-testid="stCameraInput"]:not(:has(img)) [data-testid="stCameraInputButton"]:disabled {
    opacity: .35;
}
[data-testid="stDialog"] [role="dialog"] { border-radius: 24px; border: none;
                                           background: var(--surface); }
[data-testid="stPopoverBody"] { border-radius: 18px; border: 0.5px solid var(--line);
                                background: var(--surface);
                                box-shadow: 0 16px 48px rgba(0,0,0,0.2); }
[data-testid="stToast"] { border-radius: 18px; border: 0.5px solid var(--line);
                          box-shadow: 0 12px 36px rgba(0,0,0,0.18); }

/* ---- Swipeable horizontal strips ---- */
.st-key-recent, .st-key-thumbs, .st-key-filters, .st-key-carousel {
    flex-wrap: nowrap !important; overflow-x: auto; overflow-y: hidden;
    scroll-snap-type: x mandatory; scrollbar-width: none;
    -webkit-overflow-scrolling: touch; padding: 5px 5px 8px;
}
.st-key-carousel { padding: 6px 5px 26px; margin-bottom: -14px; }
.st-key-recent::-webkit-scrollbar, .st-key-thumbs::-webkit-scrollbar,
.st-key-filters::-webkit-scrollbar, .st-key-carousel::-webkit-scrollbar { display:none; }
.st-key-recent > *, .st-key-thumbs > *, .st-key-filters > *, .st-key-carousel > * {
    flex-shrink: 0 !important; scroll-snap-align: start;
}
.st-key-carousel > * { scroll-snap-align: center; }

/* Tiles are pre-cropped to a fixed aspect in Python, so no forced heights */
[class*="st-key-recent_"] img, [class*="st-key-thumb_"] img {
    border-radius: 9px; border: 0.5px solid var(--line); background: var(--surface);
}
[class*="st-key-filt_"] img {
    border-radius: 12px; border: 0.5px solid var(--line); background: var(--surface);
}
/* Selected page / filter: ring around the image, label in brand colour */
[class*="st-key-thumb_"][class*="_sel"] img,
[class*="st-key-filt_"][class*="_sel"] img {
    outline: 2.5px solid var(--brand); outline-offset: 2px;
}
/* Whole tile is one tap target: an invisible button is stretched over it */
[class*="st-key-thumb_"], [class*="st-key-filt_"] {
    position: relative; transition: transform 0.2s var(--ease);
}
[class*="st-key-thumb_"]:active, [class*="st-key-filt_"]:active { transform: scale(0.95); }
[class*="st-key-thumb_"] [data-testid="stElementContainer"]:has(> [data-testid="stButton"]),
[class*="st-key-filt_"] [data-testid="stElementContainer"]:has(> [data-testid="stButton"]) {
    position: absolute !important; inset: 0; z-index: 3; margin: 0 !important;
    width: 100% !important; height: 100% !important;
}
[class*="st-key-thumb_"] [data-testid="stButton"],
[class*="st-key-filt_"] [data-testid="stButton"] { height: 100%; }
[class*="st-key-thumb_"] [data-testid="stButton"] button,
[class*="st-key-filt_"] [data-testid="stButton"] button {
    width: 100% !important; height: 100% !important; opacity: 0; cursor: pointer;
}
[class*="st-key-thumb_"]:has([data-testid="stButton"] button:focus-visible) img,
[class*="st-key-filt_"]:has([data-testid="stButton"] button:focus-visible) img {
    outline: 2.5px dashed var(--brand); outline-offset: 2px;
}
/* Streamlit pulls markdown blocks up by 1rem, which clipped these labels */
[data-testid="stMarkdownContainer"]:has(> .tile-label),
[data-testid="stMarkdownContainer"]:has(> .page-no) { margin-bottom: 0 !important; }
.tile-label { text-align:center; font-size:0.78rem; color:var(--muted);
              margin-top:6px; line-height:1.2; font-weight:500; }
[class*="_sel"] .tile-label { color: var(--brand); font-weight: 600; }
[class*="st-key-card_"] img { border-radius: 12px; }
.page-no { text-align:center; font-size:0.78rem !important; color:var(--muted) !important;
           font-weight:500; margin:4px 0 0 !important; }

/* ---- Page preview stage ---- */
.st-key-preview { background: var(--stage); border-radius: 22px; padding: 16px; }
.st-key-preview [data-testid="stFullScreenFrame"] {
    display:flex; justify-content:center;
}
.st-key-preview img {
    max-height: 38vh; width:auto !important; max-width:100%; margin:0 auto;
    display:block; object-fit:contain; border-radius:6px; background:#fff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1), 0 14px 34px rgba(0,0,0,0.2);
}

/* ---- Pager ---- */
.pager-label { font-weight:700; color:var(--ink); font-size:1.5rem;
               letter-spacing:-0.02em; padding-left:2px; }
.pager-label span { color:var(--muted); font-weight:400; font-size:1.05rem; }
.st-key-pager [data-testid="stPopoverButton"] > div > div:nth-child(2) { display:none; }
.st-key-pager button { width: 36px; min-width: 36px; height: 36px; min-height: 36px;
                       padding: 0; border-radius: 50% !important; }

/* ---- Empty state ---- */
.empty { text-align:center; padding: 34px 12px 16px; animation: rise 0.45s var(--ease) both; }
.empty svg { width: 96px; height: 96px; }
.empty .e-bg { fill: var(--tint-a); }
.empty .e-page { fill: var(--surface); stroke: var(--tint-b); }
.empty .e-fold { fill: var(--tint-c); }
.empty .e-l1 { fill: var(--tint-b); }
.empty .e-l2 { fill: var(--tint-c); }
.empty .e-frame { stroke: var(--brand); }
.empty h3 { margin: 16px 0 6px !important; padding: 0 !important; font-size: 1.35rem; font-weight:700; color: var(--ink);
            letter-spacing:-0.02em; }
.empty p { color: var(--muted); font-size: 0.94rem; margin: 0 auto; max-width: 300px; }

/* ---- Tips ---- */
.tips { background: var(--surface); border-radius: 16px; padding: 14px 18px; }
.tips h4 { margin:0 0 6px; font-size:0.94rem; font-weight:600; color:var(--ink); }
.tips ul { margin:0; padding-left: 1.1rem; color:var(--muted); font-size:0.88rem; }
.tips li { margin: 4px 0; }

/* ---- Quality ---- */
.q-head { display:flex; align-items:center; gap:18px; }
.ring { --p:0; --c:var(--brand); width:104px; height:104px; border-radius:50%;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--fill) 0);
        display:grid; place-items:center; flex-shrink:0; }
.ring-inner { width:82px; height:82px; border-radius:50%; background:var(--bg);
              display:flex; flex-direction:column; align-items:center; justify-content:center; }
.ring-inner b { font-size:1.7rem; color:var(--ink); line-height:1; letter-spacing:-0.02em; }
.ring-inner small { color:var(--muted); font-size:0.7rem; }
.q-label { font-size:1.2rem; font-weight:700; letter-spacing:-0.02em; }
.metric { margin: 12px 0 0; }
.metric-top { display:flex; justify-content:space-between; font-size:0.88rem; color:var(--ink); }
.metric-top span:last-child { color:var(--muted); font-variant-numeric: tabular-nums; }
.bar { height:6px; background:var(--fill); border-radius:999px; overflow:hidden; margin-top:6px; }
.bar i { display:block; height:100%; border-radius:999px; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


# ==========================================================================
# State
# ==========================================================================
def init_state() -> None:
    defaults = {
        "pages": [], "page_counter": 0, "camera_nonce": 0, "upload_nonce": 0,
        "detections": {}, "edits": {}, "versions": {}, "render_cache": {},
        "geo_cache": {}, "quality_cache": {}, "ocr_results": {}, "cur": 0,
        "nav": "scan",
        "scope": SCOPE_PAGE, "capture_mode": "Camera", "doc_name": "Scan",
        "export_format": "PDF", "ocr_lang": "eng",
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)


def prime(widget_key: str, store_key: str) -> None:
    """Seed a widget from a persistent store (widget state is dropped by
    Streamlit whenever the widget isn't on screen)."""
    if widget_key not in ss:
        ss[widget_key] = ss[store_key]


def sync(widget_key: str, store_key: str) -> None:
    ss[store_key] = ss[widget_key]


def pages() -> list:
    return ss["pages"]


def current_page():
    if not pages():
        return None
    ss["cur"] = max(0, min(ss["cur"], len(pages()) - 1))
    return pages()[ss["cur"]]


def version(page_id: int) -> int:
    return ss["versions"].get(page_id, 0)


def bump(page_id: int) -> None:
    """Invalidate a page's widgets so they reload from its stored settings."""
    ss["versions"][page_id] = version(page_id) + 1


def get_detection(page):
    if page["id"] not in ss["detections"]:
        ss["detections"][page["id"]] = detection.detect_document_corners(page["image"])
    return ss["detections"][page["id"]]


def corners_for(page):
    det = get_detection(page)
    return det.corners if det.success else None


def settings_for(page) -> EditSettings:
    """Stored edit settings; auto-crop starts on when page edges were found."""
    stored = ss["edits"].get(page["id"])
    if stored is None:
        stored = EditSettings(crop=get_detection(page).success)
    elif not isinstance(stored, EditSettings):
        # Settings created by an older version of the app (e.g. the session
        # outlived a code update): carry over known fields, default the rest.
        stored = EditSettings(**{
            f.name: getattr(stored, f.name)
            for f in dataclasses.fields(EditSettings) if hasattr(stored, f.name)
        })
    ss["edits"][page["id"]] = stored
    return stored


def _remember(cache: dict, key, value):
    if len(cache) >= CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def geometry_preview(page, settings: EditSettings):
    """Cropped/straightened/rotated page at preview size. Cached separately so
    colour and filter changes don't repeat the full-resolution geometry work."""
    key = (page["id"], settings.geometry_key())
    cache = ss["geo_cache"]
    if key in cache:
        return cache[key]
    geo = pipeline.apply_geometry(page["image"], settings, corners_for(page))
    return _remember(cache, key, pipeline.downscale(geo, PREVIEW_SIZE))


def rendered(page, settings: EditSettings, size: int):
    """Edited page rendered at ``size`` (cached by settings)."""
    key = (page["id"], settings.key(), size)
    cache = ss["render_cache"]
    if key in cache:
        return cache[key]
    base = pipeline.downscale(geometry_preview(page, settings), size)
    return _remember(cache, key, pipeline.apply_look(base, settings))


def display(page, settings: EditSettings, size: int):
    return image_io.to_display(rendered(page, settings, size))


def tile(page, settings: EditSettings, size: int, ratio: float = 1.3):
    """Thumbnail cropped to a fixed portrait aspect (height = ratio * width),
    anchored at the top of the page, so tiles line up in a strip."""
    image = display(page, settings, size)
    h, w = image.shape[:2]
    if h > ratio * w:
        return image[: int(ratio * w)]
    target_w = int(h / ratio)
    left = (w - target_w) // 2
    return image[:, left:left + target_w]


def dark_mode() -> bool:
    """Whether the viewer's browser is showing the dark theme."""
    try:
        return st.context.theme.type == "dark"
    except Exception:  # no browser context (e.g. headless tests)
        return False


def letterboxed(page, settings: EditSettings, size: int, ratio: float = 1.3):
    """Whole page centred on a neutral backdrop of fixed aspect, so cards in
    the export carousel line up without cropping any content."""
    image = display(page, settings, size)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    h, w = image.shape[:2]
    canvas_w, canvas_h = (w, int(w * ratio)) if h < w * ratio else (int(h / ratio), h)
    canvas = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (44, 44, 46) if dark_mode() else (242, 242, 247)
    top, left = (canvas_h - h) // 2, (canvas_w - w) // 2
    canvas[top:top + h, left:left + w] = image
    return canvas


def quality_report(page, settings: EditSettings):
    key = (page["id"], settings.key())
    cache = ss["quality_cache"]
    if key in cache:
        return cache[key]
    measured = pipeline.measurement_settings(settings)
    report = quality.calculate_quality_score(
        rendered(page, measured, PREVIEW_SIZE), get_detection(page).success
    )
    return _remember(cache, key, report)


@st.cache_data(ttl=120, show_spinner=False)
def ocr_status() -> tuple[bool, str, list[str]]:
    available, info = ocr.is_available()
    return available, info, ocr.available_languages() if available else []


# ==========================================================================
# Actions (used as widget callbacks)
# ==========================================================================
def go(target: str) -> None:
    ss["nav"] = target


def open_page(index: int, target: str = "edit") -> None:
    ss["cur"] = index
    ss["nav"] = target


def select_page(index: int) -> None:
    ss["cur"] = max(0, min(index, len(pages()) - 1))


def decode_and_validate(data: bytes):
    size_check = validation.validate_file_size(len(data))
    if not size_check.ok:
        return None, size_check.message
    image = image_io.decode_image(data)
    array_check = validation.validate_image_array(image)
    if not array_check.ok:
        return None, array_check.message
    return image_io.limit_resolution(image, validation.MAX_STORED_DIMENSION), ""


def add_page(image, source: str) -> None:
    ss["page_counter"] += 1
    pages().append({"id": ss["page_counter"], "image": image, "source": source})


def on_camera(widget_key: str) -> None:
    photo = ss.get(widget_key)
    if photo is None:
        return
    image, error = decode_and_validate(photo.getvalue())
    if error:
        st.toast(error, icon=":material/error:")
        return
    add_page(image, "camera")
    ss["camera_nonce"] += 1  # fresh camera for the next page
    st.toast(f"Page {len(pages())} added", icon=":material/check_circle:")


def on_upload(widget_key: str) -> None:
    files = ss.get(widget_key) or []
    added = 0
    for upload in files:
        problem = validation.validate_extension(upload.name)
        image, error = (None, problem.message) if not problem.ok else \
            decode_and_validate(upload.getvalue())
        if error:
            st.toast(f"{upload.name}: {error}", icon=":material/error:")
            continue
        add_page(image, "upload")
        added += 1
    if files:
        ss["upload_nonce"] += 1
    if added:
        noun = "page" if added == 1 else "pages"
        st.toast(f"{added} {noun} added", icon=":material/check_circle:")


def edit_targets() -> list:
    return pages() if ss["scope"] == SCOPE_ALL else [current_page()]


def set_field(field: str, value) -> None:
    current = current_page()
    for page in edit_targets():
        settings = settings_for(page)
        page_value = value
        if field == "crop":  # only pages with detected edges can auto-crop
            page_value = bool(value) and corners_for(page) is not None
        setattr(settings, field, page_value)
        if page["id"] != current["id"]:
            bump(page["id"])


def on_widget(field: str, widget_key: str) -> None:
    set_field(field, ss[widget_key])


def rotate(delta: int) -> None:
    for page in edit_targets():
        settings = settings_for(page)
        settings.rotate = (settings.rotate + delta) % 360


def reset_current() -> None:
    page = current_page()
    ss["edits"][page["id"]] = EditSettings(crop=get_detection(page).success)
    bump(page["id"])
    st.toast("Edits reset", icon=":material/restart_alt:")


def copy_look_to_all() -> None:
    page = current_page()
    base = settings_for(page)
    for other in pages():
        if other["id"] == page["id"]:
            continue
        copied = base.copy()
        copied.crop = base.crop and corners_for(other) is not None
        ss["edits"][other["id"]] = copied
        bump(other["id"])
    st.toast("Look applied to all pages", icon=":material/done_all:")


def move_page(page_id: int, direction: int) -> None:
    items = pages()
    index = next((i for i, p in enumerate(items) if p["id"] == page_id), None)
    if index is None:
        return
    target = index + direction
    if 0 <= target < len(items):
        items[index], items[target] = items[target], items[index]
        if ss["cur"] == index:
            ss["cur"] = target
        elif ss["cur"] == target:
            ss["cur"] = index


def delete_page(page_id: int) -> None:
    index = next((i for i, p in enumerate(pages()) if p["id"] == page_id), None)
    if index is None:
        return
    ss["pages"] = [p for p in pages() if p["id"] != page_id]
    for store in ("detections", "edits", "versions", "ocr_results"):
        ss[store].pop(page_id, None)
    for store in ("render_cache", "geo_cache", "quality_cache"):
        ss[store] = {k: v for k, v in ss[store].items() if k[0] != page_id}
    if ss["cur"] >= index and ss["cur"] > 0:
        ss["cur"] -= 1
    st.toast("Page deleted", icon=":material/delete:")


def clear_document() -> None:
    for store in ("detections", "edits", "versions", "ocr_results",
                  "render_cache", "geo_cache", "quality_cache"):
        ss[store] = {}
    ss["pages"] = []
    ss["cur"] = 0


# ==========================================================================
# Shared UI pieces
# ==========================================================================
def top_bar() -> None:
    count = len(pages())
    pill = (f'<span class="pill">{count} page{"" if count == 1 else "s"}</span>'
            if count else "")
    st.markdown(
        f'<div class="topbar"><div class="brand">{LOGO_SVG}'
        f'<span>Smart Document Scanner</span></div>{pill}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, text: str) -> None:
    st.markdown(
        f'<div class="empty">{EMPTY_SVG}<h3>{title}</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )
    st.button("Scan a page", type="primary", icon=":material/document_scanner:",
              width="stretch", on_click=go, args=("scan",))


def on_community_cloud() -> bool:
    """True unless the app is clearly running locally (or with ?embed=true).

    Anything that isn't localhost / a private LAN address is treated as hosted,
    so the tab bar is lifted even if the host name can't be read.
    """
    try:
        url = str(st.context.url or "")
    except Exception:
        return True
    if "embed=true" in url:
        return False
    host = url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    local = (host in ("localhost", "0.0.0.0", "[") or host.endswith(".local")
             or host.startswith(("127.", "10.", "192.168."))
             or re.match(r"172\.(1[6-9]|2\d|3[01])\.", host) is not None)
    return not local


def bottom_nav() -> None:
    with st.container(key="bottomnav"):
        st.segmented_control(
            "Navigation", list(NAV_LABELS), format_func=NAV_LABELS.get,
            key="nav", required=True, label_visibility="collapsed", width="stretch",
        )


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", name).strip().replace(" ", "_")
    return cleaned or "Scan"


# ==========================================================================
# Scan screen
# ==========================================================================
def screen_scan() -> None:
    st.markdown(
        '<div class="screen-head"><div class="screen-title">Add pages</div>'
        '<div class="muted">Capture pages one after another — each photo is '
        'added to your document automatically.</div></div>',
        unsafe_allow_html=True,
    )

    prime("w_capture", "capture_mode")
    st.segmented_control(
        "Capture mode", ["Camera", "Upload"], key="w_capture", required=True,
        format_func=lambda m: (":material/photo_camera: Camera" if m == "Camera"
                               else ":material/upload: Upload"),
        on_change=sync, args=("w_capture", "capture_mode"),
        label_visibility="collapsed", width="stretch",
    )

    if ss["capture_mode"] == "Camera":
        key = f"camera_{ss['camera_nonce']}"
        st.camera_input("Capture page", key=key, on_change=on_camera, args=(key,),
                        label_visibility="collapsed")
    else:
        key = f"upload_{ss['upload_nonce']}"
        st.file_uploader(
            "Upload images", type=list(validation.SUPPORTED_EXTENSIONS),
            accept_multiple_files=True, key=key, on_change=on_upload, args=(key,),
            label_visibility="collapsed",
        )

    if pages():
        st.markdown(f'<p class="section-label">In this document · {len(pages())}</p>',
                    unsafe_allow_html=True)
        with st.container(horizontal=True, key="recent", gap="small"):
            for index, page in enumerate(pages()):
                with st.container(key=f"recent_{page['id']}", width=70, gap=None):
                    st.image(tile(page, settings_for(page), THUMB_SIZE), width="stretch")
                    st.markdown(f'<p class="page-no">{index + 1}</p>',
                                unsafe_allow_html=True)
        st.button("Continue to edit", type="primary", icon=":material/arrow_forward:",
                  icon_position="right", width="stretch", on_click=go, args=("edit",))
    else:
        st.markdown(
            '<div class="tips"><h4>For a clean scan</h4><ul>'
            '<li>Place the page on a darker, contrasting surface</li>'
            '<li>Use even lighting and avoid casting shadows</li>'
            '<li>Keep all four corners of the page in the frame</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )


# ==========================================================================
# Edit screen
# ==========================================================================
def screen_edit() -> None:
    if not pages():
        empty_state("No pages yet", "Scan or upload a page to start editing.")
        return

    page = current_page()
    index, total = ss["cur"], len(pages())
    settings = settings_for(page)
    det = get_detection(page)

    def wkey(field: str) -> str:
        return f"w_{field}_{page['id']}_{version(page['id'])}"

    # --- Pager -----------------------------------------------------------
    with st.container(horizontal=True, key="pager", vertical_alignment="center",
                      gap="small"):
        st.markdown(f'<div class="pager-label">Page {index + 1} '
                    f'<span>of {total}</span></div>',
                    unsafe_allow_html=True, width="stretch")
        st.button("", icon=":material/chevron_left:", key="prev_page",
                  help="Previous page", disabled=index == 0,
                  on_click=select_page, args=(index - 1,))
        st.button("", icon=":material/chevron_right:", key="next_page",
                  help="Next page", disabled=index == total - 1,
                  on_click=select_page, args=(index + 1,))
        with st.popover("", icon=":material/more_vert:", help="Page options"):
            st.button("Apply this look to all pages", icon=":material/done_all:",
                      width="stretch", disabled=total == 1, on_click=copy_look_to_all)
            st.button("Move page left", icon=":material/arrow_back:", width="stretch",
                      disabled=index == 0, on_click=move_page, args=(page["id"], -1))
            st.button("Move page right", icon=":material/arrow_forward:",
                      width="stretch", disabled=index == total - 1,
                      on_click=move_page, args=(page["id"], 1))
            st.button("Reset edits", icon=":material/restart_alt:", width="stretch",
                      on_click=reset_current)
            st.button("Delete page", icon=":material/delete:", width="stretch",
                      on_click=delete_page, args=(page["id"],))

    # --- Preview ---------------------------------------------------------
    with st.container(key="preview"):
        st.image(display(page, settings, PREVIEW_SIZE), width="stretch")

    # --- Swipeable page strip ----------------------------------------------
    if total > 1:
        with st.container(horizontal=True, key="thumbs", gap="small"):
            for i, other in enumerate(pages()):
                selected = "_sel" if i == index else ""
                with st.container(key=f"thumb_{other['id']}{selected}", width=58,
                                  gap=None):
                    st.image(tile(other, settings_for(other), THUMB_SIZE),
                             width="stretch")
                    st.markdown(f'<div class="tile-label">{i + 1}</div>',
                                unsafe_allow_html=True)
                    # Invisible button stretched over the whole tile (see CSS).
                    st.button(f"Page {i + 1}", key=f"tbtn_{other['id']}",
                              type="tertiary", on_click=select_page, args=(i,))

        prime("w_scope", "scope")
        st.segmented_control(
            "Apply edits to", [SCOPE_PAGE, SCOPE_ALL], key="w_scope", required=True,
            format_func=lambda s: (":material/description: Edit this page"
                                   if s == SCOPE_PAGE
                                   else ":material/library_books: Edit all pages"),
            on_change=sync, args=("w_scope", "scope"),
            label_visibility="collapsed", width="stretch",
        )
    else:
        ss["scope"] = SCOPE_PAGE

    # --- Tools -----------------------------------------------------------
    t_filter, t_adjust, t_crop, t_text, t_quality = st.tabs(
        [":material/auto_awesome: Filters", ":material/tune: Adjust",
         ":material/crop: Crop", ":material/text_fields: Text",
         ":material/insights: Quality"],
        key="edit_tabs",
    )
    with t_filter:
        filters_panel(page, settings, wkey)
    with t_adjust:
        adjust_panel(settings, wkey)
    with t_crop:
        crop_panel(page, settings, det, wkey)
    with t_text:
        text_panel(page, settings)
    with t_quality:
        quality_panel(page, settings)


def filters_panel(page, settings: EditSettings, wkey) -> None:
    with st.container(horizontal=True, key="filters", gap="small"):
        for i, mode in enumerate(pipeline.FILTER_MODES):
            variant = settings.copy()
            variant.filter_mode = mode
            selected = "_sel" if settings.filter_mode == mode else ""
            with st.container(key=f"filt_{i}{selected}", width=80, gap=None):
                st.image(tile(page, variant, TILE_SIZE), width="stretch")
                st.markdown(f'<div class="tile-label">{FILTER_LABELS[mode]}</div>',
                            unsafe_allow_html=True)
                st.button(FILTER_LABELS[mode], key=f"fbtn_{i}", type="tertiary",
                          on_click=set_field, args=("filter_mode", mode))

    if settings.filter_mode == pipeline.FILTER_BW:
        st.slider("Detail", 3, 51, step=2, value=settings.bw_block_size,
                  key=wkey("bw_block_size"), on_change=on_widget,
                  args=("bw_block_size", wkey("bw_block_size")),
                  help="Smaller values keep finer detail.")
        st.slider("Threshold", -20, 30, value=settings.bw_c, key=wkey("bw_c"),
                  on_change=on_widget, args=("bw_c", wkey("bw_c")),
                  help="Higher values make the page whiter.")


def adjust_panel(settings: EditSettings, wkey) -> None:
    st.slider("Brightness", -100, 100, value=settings.brightness,
              key=wkey("brightness"), on_change=on_widget,
              args=("brightness", wkey("brightness")))
    st.slider("Contrast", 0.5, 2.0, step=0.05, value=float(settings.contrast),
              key=wkey("contrast"), on_change=on_widget,
              args=("contrast", wkey("contrast")))
    if settings.filter_mode != pipeline.FILTER_BW:
        st.slider("Sharpness", 0.0, 3.0, step=0.1, value=float(settings.sharpen),
                  key=wkey("sharpen"), on_change=on_widget,
                  args=("sharpen", wkey("sharpen")))
    st.segmented_control(
        "Noise reduction", [0, 1, 2], default=settings.denoise,
        format_func=lambda v: ["Off", "Light", "Strong"][v], required=True,
        key=wkey("denoise"), on_change=on_widget, args=("denoise", wkey("denoise")),
        width="stretch",
    )
    st.toggle("Clean background", value=settings.clean_background,
              key=wkey("clean_background"), on_change=on_widget,
              args=("clean_background", wkey("clean_background")),
              help="Removes shadows and evens out lighting across the page.")


def crop_panel(page, settings: EditSettings, det, wkey) -> None:
    if det.success:
        st.badge("Page edges detected", icon=":material/check_circle:", color="green")
    else:
        st.badge("Page edges not found — use manual trim", icon=":material/info:",
                 color="orange")

    st.toggle("Crop to page edges", value=settings.crop and det.success,
              disabled=not det.success, key=wkey("crop"),
              on_change=on_widget, args=("crop", wkey("crop")))
    st.toggle("Straighten", value=settings.deskew, key=wkey("deskew"),
              on_change=on_widget, args=("deskew", wkey("deskew")),
              help="Detects a slight tilt and levels the page.")

    with st.container(horizontal=True, gap="small"):
        st.button("Rotate left", icon=":material/rotate_left:", width="stretch",
                  key="rotate_left", on_click=rotate, args=(90,))
        st.button("Rotate right", icon=":material/rotate_right:", width="stretch",
                  key="rotate_right", on_click=rotate, args=(-90,))

    with st.expander("Manual trim", icon=":material/crop_free:",
                     expanded=not det.success):
        for field, label in (("margin_top", "Top"), ("margin_bottom", "Bottom"),
                             ("margin_left", "Left"), ("margin_right", "Right")):
            st.slider(label, 0, 40, value=getattr(settings, field), format="%d%%",
                      key=wkey(field), on_change=on_widget, args=(field, wkey(field)))

    with st.expander("How this page was detected", icon=":material/visibility:"):
        small, _ = preprocessing.resize_for_detection(page["image"], 700)
        st.image(preprocessing.to_grayscale(small), caption="Grayscale", width="stretch")
        if det.edges is not None:
            st.image(det.edges, caption="Edge map", width="stretch")
        if det.success:
            outlined = detection.draw_corners(page["image"], det.corners, thickness=8)
            st.image(image_io.to_display(preprocessing.resize_for_detection(outlined, 700)[0]),
                     caption="Detected page outline", width="stretch")
        else:
            st.caption(det.message)


def text_panel(page, settings: EditSettings) -> None:
    available, _, languages = ocr_status()
    if not available:
        st.markdown('<p class="muted">Text recognition isn’t available yet. '
                    'It needs the free Tesseract OCR engine installed where '
                    'the app runs.</p>', unsafe_allow_html=True)
        with st.expander("How to enable text recognition", icon=":material/help:"):
            st.markdown(
                "1. Download the Windows installer from the "
                "[UB Mannheim Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki).\n"
                "2. Install it (default folder: `C:\\Program Files\\Tesseract-OCR`).\n"
                "3. Add that folder to your system **PATH**.\n"
                "4. Restart the app."
            )
            if st.button("Check again", icon=":material/refresh:"):
                ocr_status.clear()
                st.rerun()
        return

    languages = languages or ["eng"]
    if ss["ocr_lang"] not in languages:
        ss["ocr_lang"] = "eng" if "eng" in languages else languages[0]
    prime("w_ocr_lang", "ocr_lang")

    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        st.selectbox("Language", languages, key="w_ocr_lang", on_change=sync,
                     args=("w_ocr_lang", "ocr_lang"), width="stretch")
        run = st.button("Recognize", type="primary", icon=":material/text_fields:")

    if run:
        with st.spinner("Recognizing text…"):
            full = pipeline.render_page(page["image"], settings, corners_for(page))
            result = ocr.extract_text_with_confidence(full, ss["ocr_lang"])
        ss["ocr_results"][page["id"]] = (settings.key(), ss["ocr_lang"], result)

    stored = ss["ocr_results"].get(page["id"])
    if stored is None:
        st.markdown('<p class="muted small">Tip: the B&W filter usually gives the '
                    'most accurate text.</p>', unsafe_allow_html=True)
        return

    key, lang, result = stored
    if key != settings.key() or lang != ss["ocr_lang"]:
        st.markdown('<p class="muted small">The page changed since the last '
                    'recognition. Tap Recognize to update.</p>', unsafe_allow_html=True)
        return
    if not result.success:
        st.warning(result.message, icon=":material/error:")
        return
    if not result.text:
        st.info(result.message, icon=":material/info:")
        return

    st.text_area("Recognized text", value=result.text, height=170,
                 key=f"ocr_text_{page['id']}_{hash(key)}", label_visibility="collapsed")
    meta = f"{result.word_count} words"
    if result.mean_confidence is not None:
        meta += f" · {result.mean_confidence:.0f}% recognition confidence"
    st.markdown(f'<p class="muted small">{meta}</p>', unsafe_allow_html=True)
    st.download_button("Download text", data=result.text.encode("utf-8"),
                       file_name=f"{safe_filename(ss['doc_name'])}_page_{ss['cur'] + 1}.txt",
                       mime="text/plain", icon=":material/download:", width="stretch",
                       on_click="ignore", key=f"ocr_dl_{page['id']}")


def quality_panel(page, settings: EditSettings) -> None:
    report = quality_report(page, settings)
    if report is None:
        st.info("Quality could not be measured for this page.")
        return

    color = QUALITY_COLORS[report.label]
    score = round(report.overall)
    summary = {
        "Excellent": "Crisp, well-exposed and clean.",
        "Good": "Clear and readable.",
        "Fair": "Readable, but could be improved.",
        "Poor": "Hard to read — try retaking the photo.",
    }[report.label]

    metrics = [
        ("Sharpness", report.scores["sharpness"]),
        ("Contrast", report.scores["contrast"]),
        ("Exposure", report.scores["brightness"]),
        ("Cleanliness", report.scores["noise"]),
    ]
    bars = "".join(
        f'<div class="metric"><div class="metric-top"><span>{name}</span>'
        f'<span>{value:.0f}</span></div><div class="bar"><i style="width:{value:.0f}%;'
        f'background:{color}"></i></div></div>'
        for name, value in metrics
    )
    st.markdown(
        f'<div class="q-head"><div class="ring" style="--p:{score};--c:{color}">'
        f'<div class="ring-inner"><b>{score}</b><small>of 100</small></div></div>'
        f'<div><div class="q-label" style="color:{color}">{report.label}</div>'
        f'<p class="muted">{summary}</p></div></div>{bars}',
        unsafe_allow_html=True,
    )

    tips = []
    if report.scores["sharpness"] < 50:
        tips.append("Hold the camera steady or increase Sharpness.")
    if report.scores["brightness"] < 60:
        tips.append("Adjust Brightness, or turn on Clean background.")
    if report.scores["contrast"] < 50:
        tips.append("Try the Magic or B&W filter for stronger contrast.")
    if report.scores["noise"] < 50:
        tips.append("Turn on Noise reduction.")
    if tips:
        st.markdown('<p class="section-label" style="margin-top:14px">Suggestions</p>'
                    + "".join(f'<p class="muted small">• {t}</p>' for t in tips),
                    unsafe_allow_html=True)

    with st.expander("How the score works", icon=":material/info:"):
        st.markdown(
            f"Raw measurements: brightness {report.brightness:.0f}, contrast "
            f"{report.contrast:.1f}, sharpness {report.sharpness:.0f}, noise "
            f"{report.noise:.1f}.\n\n"
            "Each is scaled to 0–100 and weighted (sharpness 35%, contrast 25%, "
            "exposure 20%, cleanliness 20%). Black & white pages are measured "
            "on their grayscale version, since pure black and white would "
            "inflate the result. The score is an estimate based on image "
            "statistics and is a guide, not an absolute measure."
        )


# ==========================================================================
# Export screen
# ==========================================================================
def _render_all(snapshot):
    return [pipeline.render_page(img, s, c) for img, s, c in snapshot]


def _build_pdf(snapshot) -> bytes:
    return document.images_to_pdf_bytes(_render_all(snapshot)) or b""


def _build_zip(snapshot) -> bytes:
    return document.images_to_zip_bytes(_render_all(snapshot), ".png") or b""


def _build_text(snapshot, language: str) -> bytes:
    parts = []
    for number, image in enumerate(_render_all(snapshot), start=1):
        result = ocr.extract_text(image, language)
        body = result.text if result.success and result.text else "[No text found]"
        parts.append(f"--- Page {number} ---\n{body}")
    return "\n\n".join(parts).encode("utf-8")


def start_new_document() -> None:
    clear_document()
    ss["nav"] = "scan"
    st.toast("Ready for a new document", icon=":material/add:")


@st.dialog("Start a new document?")
def confirm_new_document() -> None:
    st.write("All pages and edits in the current document will be removed.")
    with st.container(horizontal=True, gap="small"):
        if st.button("Cancel", width="stretch"):
            st.rerun()
        # The callback does the work, so it runs whether Streamlit reruns just
        # the dialog or the whole app; st.rerun() then closes the dialog.
        if st.button("Remove pages", type="primary", width="stretch",
                     on_click=start_new_document):
            st.rerun()


def screen_export() -> None:
    if not pages():
        empty_state("Nothing to export", "Scan or upload pages to build your document.")
        return

    total = len(pages())
    st.markdown(
        f'<div class="screen-head"><div class="screen-title">Your document</div>'
        f'<div class="muted">{total} page{"" if total == 1 else "s"}'
        f'{" · swipe to review" if total > 1 else ""}</div></div>',
        unsafe_allow_html=True,
    )

    with st.container(horizontal=True, key="carousel", gap="medium"):
        for i, page in enumerate(pages()):
            settings = settings_for(page)
            report = quality_report(page, settings)
            with st.container(key=f"card_{page['id']}", width=250, border=True,
                              gap="small"):
                st.image(letterboxed(page, settings, CARD_SIZE), width="stretch")
                with st.container(horizontal=True, vertical_alignment="center",
                                  horizontal_alignment="distribute"):
                    st.markdown(f"**Page {i + 1}**")
                    if report is not None:
                        st.badge(f"{report.label} · {report.overall:.0f}",
                                 color={"Excellent": "green", "Good": "blue",
                                        "Fair": "orange", "Poor": "red"}[report.label])
                with st.container(horizontal=True, horizontal_alignment="distribute"):
                    st.button("", icon=":material/arrow_back:", key=f"cl_{page['id']}",
                              help="Move left", type="tertiary", disabled=i == 0,
                              on_click=move_page, args=(page["id"], -1))
                    st.button("", icon=":material/edit:", key=f"ce_{page['id']}",
                              help="Edit page", type="tertiary",
                              on_click=open_page, args=(i,))
                    st.button("", icon=":material/delete:", key=f"cd_{page['id']}",
                              help="Delete page", type="tertiary",
                              on_click=delete_page, args=(page["id"],))
                    st.button("", icon=":material/arrow_forward:", key=f"cr_{page['id']}",
                              help="Move right", type="tertiary", disabled=i == total - 1,
                              on_click=move_page, args=(page["id"], 1))

    available, _, _ = ocr_status()
    formats = ["PDF", "Images"] + (["Text"] if available else [])
    if ss["export_format"] not in formats:
        ss["export_format"] = "PDF"

    with st.container(border=True, key="export_panel"):
        prime("w_doc_name", "doc_name")
        st.text_input("File name", key="w_doc_name", on_change=sync,
                      args=("w_doc_name", "doc_name"), max_chars=60)
        prime("w_format", "export_format")
        st.segmented_control(
            "Format", formats, key="w_format", required=True,
            format_func=lambda f: {"PDF": ":material/picture_as_pdf: PDF",
                                   "Images": ":material/photo_library: Images",
                                   "Text": ":material/notes: Text"}[f],
            on_change=sync, args=("w_format", "export_format"), width="stretch",
        )

        name = safe_filename(ss["doc_name"])
        snapshot = [(p["image"], settings_for(p).copy(), corners_for(p)) for p in pages()]
        fmt = ss["export_format"]
        if fmt == "PDF":
            data, file_name, mime = (lambda: _build_pdf(snapshot)), f"{name}.pdf", \
                "application/pdf"
        elif fmt == "Images":
            data, file_name, mime = (lambda: _build_zip(snapshot)), f"{name}.zip", \
                "application/zip"
        else:
            lang = ss["ocr_lang"]
            data, file_name, mime = (lambda: _build_text(snapshot, lang)), \
                f"{name}.txt", "text/plain"

        st.download_button(
            f"Download {fmt if fmt != 'Images' else 'images'}", data=data,
            file_name=file_name, mime=mime, type="primary",
            icon=":material/download:", width="stretch", on_click="ignore",
        )
        st.markdown('<p class="muted small">Exported at full resolution with all '
                    'edits applied, in the order shown above.</p>',
                    unsafe_allow_html=True)

    if st.button("Start a new document", icon=":material/add:", type="tertiary",
                 width="stretch"):
        confirm_new_document()


# ==========================================================================
# App
# ==========================================================================
def main() -> None:
    init_state()
    st.markdown(APP_CSS, unsafe_allow_html=True)
    if on_community_cloud():
        # The "Hosted with Streamlit" badge sits outside the app (bottom-right
        # of the host page) and can't be hidden, so float the tab bar above it.
        st.markdown("<style>:root { --nav-lift: 64px; }</style>",
                    unsafe_allow_html=True)
    top_bar()

    screen = {"scan": screen_scan, "edit": screen_edit, "export": screen_export}
    screen.get(ss["nav"], screen_scan)()

    bottom_nav()


main()
