# Smart Document Scanner

A mobile-first, browser-based document scanner built with Python, OpenCV and
Streamlit. Photograph or upload pages, and the app finds each page's edges,
flattens it, cleans it up with scanner-style filters, measures scan quality,
extracts the text, and exports the result as a PDF, images, or text.

> **Everything runs locally.** Documents are never sent to an external service,
> and text recognition uses a locally installed Tesseract engine.

## Using the app

The app has three screens, switched from the bottom tab bar:

| Screen | What you can do |
|--------|-----------------|
| **Scan** | Capture pages with the camera one after another (each photo is added automatically), or upload several JPG/PNG images at once. |
| **Edit** | Swipe through pages and edit them: filters (Original, Magic, Grayscale, B&W, Contrast) with live previews; brightness, contrast, sharpness, noise reduction and background cleanup; auto-crop to the page edges, straighten, rotate, manual trim; text recognition; and a quality score with suggestions. Choose **Edit this page** or **Edit all pages** to apply changes to one page or the whole document. The ⋮ menu copies a page's look to all pages, reorders, resets, or deletes pages. |
| **Export** | Swipe through the finished pages, reorder or delete them, name the file and download it as **PDF**, **images (ZIP)**, or **text** (when OCR is available). Exports are rendered at full resolution with all edits applied. |

## Image-processing techniques

| Step | Technique | Module |
|------|-----------|--------|
| Preprocessing | Grayscale conversion, Gaussian blur, resizing for detection | `scanner/preprocessing.py` |
| Page detection | Canny edges → morphological closing → contours → polygon approximation → largest convex quadrilateral | `scanner/detection.py` |
| Perspective correction | Corner ordering + homography (`getPerspectiveTransform` / `warpPerspective`) | `scanner/perspective.py` |
| Straightening | Probabilistic Hough lines → median angle → rotation with expanded canvas | `scanner/deskew.py` |
| Enhancement | Linear brightness/contrast, CLAHE on the LAB lightness channel, unsharp masking | `scanner/enhancement.py` |
| Noise & shadows | Median / Gaussian filtering; background estimation (dilation + large median blur) and division to even out lighting | `scanner/noise_removal.py` |
| Filters | Grayscale, adaptive Gaussian threshold (B&W), global / Otsu threshold, histogram equalization | `scanner/filters.py` |
| Quality score | Mean intensity, standard deviation, variance of the Laplacian, Immerkær noise estimate on flat regions → weighted 0–100 score | `scanner/quality.py` |
| OCR | Tesseract via `pytesseract`, with Otsu binarization and confidence reporting | `scanner/ocr.py` |

The **quality score** is a heuristic estimate, not a universal measure. Each
metric is scaled to 0–100 with documented thresholds and combined with weights
(sharpness 35%, contrast 25%, exposure 20%, cleanliness 20%) defined in one
place in `scanner/quality.py`. Black & white pages are measured on their
grayscale version, because pure black and white would inflate the statistics.

## Project structure

```
smart-document-scanner/
├── app.py                  # Streamlit UI only (no processing logic)
├── requirements.txt
├── .streamlit/config.toml  # Theme, upload limit, usage statistics off
├── scanner/
│   ├── preprocessing.py    # Grayscale, blur, resize, Canny
│   ├── detection.py        # Page boundary detection
│   ├── perspective.py      # Perspective correction
│   ├── deskew.py           # Skew estimation and rotation
│   ├── enhancement.py      # Brightness/contrast, CLAHE, sharpening
│   ├── noise_removal.py    # Denoising and shadow removal
│   ├── filters.py          # Scanner filters
│   ├── ocr.py              # Text recognition
│   ├── quality.py          # Scan quality metrics and score
│   └── pipeline.py         # EditSettings + per-page render pipeline
├── utils/
│   ├── image_io.py         # Decode / encode / colour conversion
│   ├── document.py         # PDF and ZIP assembly
│   ├── geometry.py         # Corner ordering, quad geometry
│   └── validation.py       # Input validation
└── tests/                  # Unit tests + headless UI tests (Streamlit AppTest)
```

## Setup (Windows PowerShell)

1. Install **Python 3.10+**.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. *(Optional, for text recognition)* Install the **Tesseract OCR engine**.
   `pytesseract` is only a Python wrapper — it does not include the engine or
   its language data. Download the Windows installer from the
   [UB Mannheim Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki),
   install it (default `C:\Program Files\Tesseract-OCR`), add that folder to
   your **PATH**, and restart the app. Everything else works without it; the
   Text tab shows these steps until the engine is found.

## Run

```powershell
streamlit run app.py
```

Browsers only allow camera access on `localhost` or over HTTPS. To use the
camera from a phone on the same network, serve the app over HTTPS, or use the
**Upload** option — on most phones it can open the camera directly.

## Tests

```powershell
pytest -q
```

The suite covers the processing modules with synthetic images and drives the
UI headlessly (every screen, filters, edit scope, crop tools, page management,
export, and the new-document flow).

## Limitations

- Designed for one document per photo, on a background that contrasts with
  the paper. Similar colours, cut-off corners, or heavy folds can defeat edge
  detection — use **Manual trim** in that case.
- Very dark or blurred photos reduce detection and OCR accuracy.
- OCR accuracy depends on text size, font, language data and image quality;
  handwriting is not supported.
- The quality score is an image-statistics estimate and should be read as a
  guide.
- The original photos are never modified; all edits are applied to copies.
