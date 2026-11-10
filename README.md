# Smart Document Scanner

A mobile-first, browser-based document scanner built with Python, OpenCV and
Streamlit. Photograph or upload pages, and the app finds each page's edges,
flattens it, cleans it up with scanner-style filters, measures scan quality,
extracts the text, and exports the result as a PDF, images, or text.

> **No third-party services.** Images are processed by the app's own server
> (your computer when run locally) and are never sent to an external API; text
> recognition uses the Tesseract engine installed alongside the app.

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

3. Install dependencies (the dev file adds `pytest` on top of the app's own
   requirements):

   ```powershell
   pip install -r requirements-dev.txt
   ```

   The app uses `opencv-python-headless`. If an older setup already has
   `opencv-python` installed in the same environment, remove it first with
   `pip uninstall -y opencv-python`, since both packages provide `cv2`.

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

## Deploy (Streamlit Community Cloud)

The app needs a running Python server, so static hosts such as Netlify,
Vercel static, or GitHub Pages cannot run it. Streamlit Community Cloud is free
and deploys directly from this GitHub repository:

1. Push the latest code to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Choose **Create app → Deploy a public app from GitHub** and set:
   - **Repository:** `UpmanyuCodes/Smart-Document-Scanner`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Advanced settings → Python version:** 3.12
4. Click **Deploy**. The first build takes a few minutes.

What the platform uses from the repository:

| File | Purpose |
|------|---------|
| `requirements.txt` | Python packages (headless OpenCV for servers without a display) |
| `packages.txt` | System packages installed with apt: the Tesseract engine and English language data, so text recognition works online |
| `.streamlit/config.toml` | Theme, 40 MB upload limit, usage statistics off |

The hosted app is served over HTTPS, so the in-app camera also works on
phones. Uploaded pages stay in the server's memory for the browser session
only; they are not saved to disk. Pages are stored at up to 3508 px on the
longest side (A4 at 300 DPI) to keep memory use within free-tier limits.

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
