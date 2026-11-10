# Smart Document Scanner

Turn a photograph of a physical document into a corrected, enhanced, readable
digital scan — then extract its text and estimate the scan quality. Built with
Python, OpenCV, NumPy, Pillow, pytesseract, and a Streamlit interface.

> **Everything runs locally.** Your documents are never uploaded to any
> external service, and OCR runs through a locally installed Tesseract engine.

## Build status

This project is being built in phases. Current state:

| Phase | Scope | Status |
|------:|-------|--------|
| 1 | Project foundation, image upload, validation & preview | ✅ Done |
| — | In-browser camera capture + multi-page document (PDF/ZIP export, reorder/delete) | ✅ Done |
| — | Mobile-app UI (phone-width frame, app bar, segmented nav, card layout) | ✅ Done |
| 2 | Preprocessing + automatic document detection (edges, contours, 4 corners) | ✅ Done |
| 3 | Perspective correction + deskewing | ✅ Done |
| 4 | Enhancement (brightness/contrast/CLAHE/sharpen) | ✅ Done |
| 5 | Document filters (grayscale / B&W / high contrast) | ✅ Done |
| — | CamScanner-style **Edit** screen: per-page filters + adjustments, apply-to-all | ✅ Done |
| 6 | OCR text extraction (Tesseract) | ⏳ Planned |
| 7 | Scan quality analyzer | ⏳ Planned |
| 8 | Full pipeline integration, visualizations, final testing | ⏳ Planned |

## Project structure

```
smart-document-scanner/
├── app.py                 # Streamlit UI (thin; no processing logic)
├── requirements.txt
├── README.md
├── .gitignore
├── scanner/               # Image-processing modules (filled in across phases)
│   ├── __init__.py
│   ├── preprocessing.py   # Grayscale, blur, resize-for-detection, Canny
│   ├── detection.py       # Edges -> contours -> 4-corner document boundary
│   ├── perspective.py     # Homography / top-down flattening
│   ├── deskew.py          # Hough-line skew estimate + rotation
│   ├── enhancement.py     # Brightness/contrast, CLAHE, unsharp sharpen
│   ├── filters.py         # Grayscale, global/adaptive/Otsu B&W, high contrast
│   └── pipeline.py        # EditSettings + per-page render pipeline
├── utils/
│   ├── __init__.py
│   ├── image_io.py        # Decode / encode / format conversion
│   ├── document.py        # Multi-page PDF / ZIP assembly
│   ├── geometry.py        # Corner ordering & quad geometry
│   └── validation.py      # Input & parameter validation
├── tests/
│   ├── test_image_io.py
│   ├── test_document.py
│   ├── test_geometry.py
│   ├── test_detection.py
│   ├── test_perspective.py
│   ├── test_filters.py
│   ├── test_pipeline.py
│   └── test_validation.py
└── assets/
```

## Setup (Windows PowerShell)

1. **Install Python** (3.10+ recommended).

2. **Create and activate a virtual environment:**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (needed for the OCR feature in Phase 6).
   `pytesseract` is only a Python wrapper — it does **not** install the
   Tesseract engine or its language data. Download the Windows installer from
   the [UB Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
   and either add its install folder (e.g.
   `C:\Program Files\Tesseract-OCR`) to your `PATH`, or configure the path in
   the app once OCR lands. The rest of the scanner works without Tesseract.

## Run the app

```powershell
streamlit run app.py
```

This opens the scanner in your browser. From the sidebar you can:

- **📷 Camera** — take a photo of a page with your webcam, then add it. Capture
  pages one after another to build a multi-page document. (The browser will ask
  for camera permission; it works over `localhost` without HTTPS.)
- **📁 Upload** — add one or several JPG/JPEG/PNG images at once.

Pages appear as a thumbnail strip you can reorder (⬆️/⬇️) or remove (🗑️). Export
the whole document as a **PDF** or a **ZIP** of images, and inspect any single
page in detail.

## Run the tests

```powershell
pytest -q
```

## Notes & limitations

- Designed for a **single document** photographed against a reasonably
  distinguishable background.
- The upcoming quality score is a **heuristic estimate** from measurable image
  properties, not a universal measure of document quality.
- The original uploaded image is always preserved; all processing runs on
  copies.
