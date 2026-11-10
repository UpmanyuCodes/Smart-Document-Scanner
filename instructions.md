# Smart Document Scanner Using Digital Image Processing

## Product Requirements, Technical Specification & Claude Code Build Instructions

> **Project type:** Image Processing and Computer Vision Lab\
> **Primary language:** Python\
> **Core libraries:** OpenCV, NumPy, Pillow, pytesseract\
> **Suggested interface:** Streamlit\
> **OCR engine:** Tesseract OCR\
> **Goal:** Build an end-to-end document scanning application that turns
> a photograph of a physical document into a corrected, enhanced,
> readable digital scan, extracts printed text, and reports an estimated
> scan-quality score.

------------------------------------------------------------------------

# 1. Instructions for Claude Code

You are an experienced Python, OpenCV, and computer-vision engineer.
Build this project as a complete, runnable, modular application.

Before writing code:

1.  Inspect the existing repository and identify its current structure,
    dependencies, and files.
2.  If the repository already contains useful code, preserve and improve
    it rather than replacing it unnecessarily.
3.  Create a short implementation plan.
4.  Implement the application in small, testable stages.
5.  Run the available tests and fix errors before considering the work
    complete.
6.  Do not leave core features as placeholders, fake buttons, or
    hard-coded demo outputs.
7.  If a feature cannot reliably work for every image, implement a
    sensible fallback and explain the limitation in the interface or
    documentation.
8.  Keep image-processing logic separate from the user interface.
9.  Add comments where the image-processing logic is non-obvious, but
    avoid excessive comments.
10. Make the project straightforward to run locally on Windows.

The result must be a usable application, not merely a collection of
example scripts.

## Important implementation rules

-   Use Python and OpenCV for the main image-processing pipeline.
-   Use NumPy for numerical operations.
-   Use Pillow only where useful for image loading or format conversion.
-   Use `pytesseract` to connect to the locally installed Tesseract OCR
    engine.
-   Use Streamlit for the initial user interface unless the existing
    repository already uses a suitable working GUI framework.
-   Do not use a cloud OCR API. OCR should run locally through
    Tesseract.
-   Do not require a GPU.
-   Do not upload user documents to an external service.
-   Keep original images unchanged. All processing must operate on
    copies.
-   Handle invalid, corrupted, unsupported, and unusually large image
    files gracefully.
-   Make processing parameters configurable where practical.
-   Avoid claiming that the quality score is an objective or universal
    measure. It is an estimated score based on measurable image
    properties.

------------------------------------------------------------------------

# 2. Project Overview

The Smart Document Scanner is an intelligent document-scanning system
that uses Digital Image Processing techniques to convert photographs of
physical documents into clean, corrected, and digitally usable
documents.

A user may photograph a sheet of paper lying on a desk. The photograph
may contain perspective distortion, rotation, uneven illumination,
shadows, noise, or low contrast. The application should identify the
document, correct its geometry, improve readability, and provide
scanner-style output.

The application must also:

-   Extract printed text using Optical Character Recognition (OCR).
-   Evaluate scan quality using measurable image properties.
-   Display the processed result and its quality analysis.
-   Allow the user to download the processed document and extracted
    text.

## Main objective

Develop an intelligent document-scanning system that applies Digital
Image Processing techniques to detect, correct, enhance, evaluate, and
extract information from photographed documents.

## Problem being addressed

Photographs of documents frequently contain:

-   Perspective distortion.
-   Uneven illumination.
-   Shadows.
-   Image noise.
-   Tilt or rotation.
-   Low contrast.
-   Blur.

Manually correcting these problems is time-consuming and can produce
inconsistent results. This project automates these operations while
demonstrating the underlying image-processing techniques.

------------------------------------------------------------------------

# 3. Target User Workflow

The expected workflow is:

1.  The user opens the application.
2.  The user uploads a photograph of a single physical document.
3.  The application displays the original image.
4.  The application preprocesses the image.
5.  The application detects the document boundary and its four corners.
6.  The application corrects perspective distortion, if a reliable
    document boundary is found.
7.  The application detects and corrects small rotations or skew.
8.  The application enhances brightness, contrast, and sharpness.
9.  The application reduces noise and attempts to reduce shadows.
10. The application generates scanner-style output modes.
11. The application calculates scan-quality indicators.
12. The application extracts printed text using OCR.
13. The user compares the original and processed images.
14. The user downloads the processed image and OCR text.

The user should be able to run individual processing operations or use
an **Auto Scan** workflow that runs the complete pipeline.

------------------------------------------------------------------------

# 4. Functional Requirements

## Feature 1: Image Capture / Upload

Allow the user to provide a document photograph.

Requirements:

-   Support JPG, JPEG, and PNG.
-   Display the uploaded image.
-   Display image dimensions.
-   Preserve the original image.
-   Validate the uploaded file before processing.
-   Show a useful error message if the file cannot be decoded.
-   Optionally support camera capture if it can be implemented reliably
    in the selected interface.

The initial implementation must support image upload. Camera capture is
an optional enhancement, not a blocker.

## Feature 2: Image Preprocessing

Prepare the input image for document detection and later operations.

Implement:

-   Color image loading.
-   BGR-to-grayscale conversion where needed.
-   Gaussian blur to reduce high-frequency noise.
-   Appropriate resizing for computationally expensive detection
    operations, while preserving enough detail.
-   A clear distinction between the detection image and the
    full-resolution image used for final output.

Show the preprocessed image in an optional intermediate-results section.

Relevant concepts:

-   Grayscale conversion.
-   Gaussian filtering.
-   Image resizing.
-   Noise reduction.

## Feature 3: Automatic Document Detection

Detect the paper/document region against the surrounding background.

Suggested approach:

1.  Convert a working copy to grayscale.
2.  Apply Gaussian blur.
3.  Apply Canny edge detection.
4.  Find contours.
5.  Sort contours by area, largest first.
6.  Approximate candidate contours using polygon approximation.
7.  Identify a likely four-sided contour.
8.  Validate that the candidate has reasonable area and geometry.
9.  Return its four corner points in a consistent order.

Use OpenCV functions such as:

-   `cv2.cvtColor`
-   `cv2.GaussianBlur`
-   `cv2.Canny`
-   `cv2.findContours`
-   `cv2.contourArea`
-   `cv2.approxPolyDP`
-   `cv2.arcLength`

Requirements:

-   Draw the detected boundary on a copy of the original image.
-   Display the four detected corners or the detected outline.
-   Provide a clear message if no reliable document boundary is found.
-   Do not silently treat an arbitrary large contour as a valid
    document.
-   Include a manual-corner fallback if feasible, so the user can still
    scan an image when automatic detection fails.

Important: contour-based detection is heuristic. It will not work
perfectly when the paper and background have similar colors, when edges
are obscured, or when the document is severely folded or cropped.

## Feature 4: Perspective Correction

Transform a document photographed at an angle into a rectangular,
top-down view.

Use a perspective transformation / homography.

Implementation outline:

1.  Receive four document corner coordinates.
2.  Order the points consistently: top-left, top-right, bottom-right,
    bottom-left.
3.  Calculate output width from the larger of the top and bottom edge
    lengths.
4.  Calculate output height from the larger of the left and right edge
    lengths.
5.  Define the destination rectangle.
6.  Calculate the perspective transform matrix using
    `cv2.getPerspectiveTransform`.
7.  Apply `cv2.warpPerspective`.
8.  Return the corrected document.

Requirements:

-   Preserve as much useful resolution as possible.
-   Prevent zero or unreasonable output dimensions.
-   Handle invalid corner coordinates.
-   Show the corrected image.
-   If automatic corner detection fails, allow the user to continue with
    the original image or use manually selected corners.

## Feature 5: Auto Rotation / Deskewing

Detect and correct small document rotations or tilted text/edges.

Suggested methods:

-   Hough Line Transform using `cv2.HoughLinesP`.
-   Estimate dominant line orientation.
-   Calculate a robust representative angle.
-   Rotate the image only when the estimated angle is within a
    reasonable correction range.

Requirements:

-   Avoid rotating an already aligned document unnecessarily.
-   Use a configurable angle threshold.
-   Expand the output canvas when needed to avoid cutting off document
    content.
-   Explain in the UI when no meaningful skew was detected.

Do not assume that every strong line in a document is horizontal. Use
sensible filtering and avoid treating arbitrary text strokes as reliable
page orientation.

## Feature 6: Image Enhancement

Improve visual readability.

Implement the following operations as separate functions:

### A. Brightness adjustment

Allow the user to adjust brightness using a slider.

### B. Contrast adjustment

Allow the user to adjust contrast using a slider.

### C. Histogram equalization

Implement grayscale histogram equalization using `cv2.equalizeHist`.

For color images, consider applying CLAHE to the LAB color space's
lightness channel rather than independently equalizing color channels.

### D. Sharpening

Implement sharpening using a convolution kernel or unsharp masking.

Requirements:

-   Keep the original image available for comparison.
-   Avoid excessive sharpening and visible artifacts.
-   Provide before/after comparison.
-   Let the user control brightness, contrast, and sharpening strength.

## Feature 7: Noise and Shadow Removal

Reduce unwanted noise and illumination artifacts.

Implement:

-   Median filtering using `cv2.medianBlur`.
-   Gaussian filtering using `cv2.GaussianBlur`.
-   A shadow-reduction method based on background/illumination
    estimation.
-   Morphological operations where appropriate.

A possible shadow-reduction approach:

1.  Convert the image to grayscale or a suitable color channel.
2.  Estimate slowly varying background illumination using a large-kernel
    blur or morphological operation.
3.  Normalize or divide the image by the estimated background, with
    safeguards against division by zero.
4.  Rescale the result to a useful intensity range.
5.  Compare the result with the original to ensure that text is not
    destroyed.

Requirements:

-   Keep noise reduction and shadow correction as separate operations.
-   Make the strength configurable.
-   Avoid aggressive processing that removes thin text or document
    details.
-   If shadow correction makes the image worse, allow the user to
    disable it.

## Feature 8: Smart Document Filters

Generate multiple scanner-style versions.

Required output modes:

1.  Original.
2.  Grayscale.
3.  Black & White.
4.  High Contrast.

For Black & White output, implement:

-   Global thresholding.
-   Adaptive thresholding, preferably adaptive Gaussian thresholding.
-   A user-adjustable threshold or block-size/C parameter where
    appropriate.

Use OpenCV functions such as:

-   `cv2.threshold`
-   `cv2.adaptiveThreshold`
-   `cv2.equalizeHist`

Requirements:

-   Preserve a readable output.
-   Show each mode separately.
-   Do not overwrite the original image.
-   Allow the user to select which output to download.

## Feature 9: OCR --- Text Extraction

Extract machine-readable text from the processed document.

Use:

-   Tesseract OCR installed locally.
-   `pytesseract` Python package.
-   OCR preprocessing where useful.

Requirements:

-   Allow OCR on the processed image.
-   Allow the user to choose a Tesseract language if the language data
    is installed.
-   Display extracted text in a readable text area.
-   Allow text download as a `.txt` file.
-   Handle missing Tesseract installation with clear setup instructions.
-   Handle missing language data with a clear message.
-   Do not pretend that OCR output is correct when the engine returns
    poor or empty results.

Optional enhancement:

-   Display OCR confidence using `pytesseract.image_to_data`.
-   If confidence is displayed, explain that it is an OCR-engine
    confidence estimate, not a guarantee of correctness.

OCR accuracy depends on text size, font, language, image quality, and
preprocessing.

## Feature 10: Scan Quality Analyzer

This is the key differentiating feature of the project.

The application must evaluate the processed scan using measurable image
properties.

Evaluate at least:

### A. Brightness

Calculate mean grayscale intensity.

A very dark or very bright image may indicate poor exposure.

### B. Contrast

Use a measurable statistic such as grayscale standard deviation or a
contrast measure based on intensity distribution.

### C. Sharpness

Use the variance of the Laplacian as a practical blur/sharpness
indicator.

Example:

`sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()`

Higher values often indicate more high-frequency detail, but text,
noise, and image content affect this measurement.

### D. Noise estimate

Implement a practical image-noise estimate. Document the chosen method
and its limitations. Do not claim that a simple metric perfectly
separates noise from document texture.

### E. Document boundary validation

Report whether a plausible document boundary was detected and corrected.

### F. Overall scan-quality score

Combine normalized metric scores into an estimated score between 0 and
100.

Requirements:

-   Each metric must be calculated from the actual processed image.
-   Normalize metrics before combining them.
-   Use documented thresholds or a documented scoring method.
-   Make the weights configurable in one place.
-   Do not hard-code a sample score such as 89%.
-   Do not generate random scores.
-   Display the overall score and individual metric results.
-   Display a simple label such as Excellent, Good, Fair, or Poor.
-   Explain that the score is a heuristic estimate, not a universal
    standard.
-   Make the scoring logic testable independently from the UI.

A suggested starting design is a weighted score:

`overall_score = sum(metric_score * metric_weight)`

The metric weights should add up to 1.0. Each metric score should be
normalized to the range 0--100.

Use a documented scoring function with reasonable ranges. Do not assume
that one threshold works equally well for every document type. Keep
thresholds easy to tune.

------------------------------------------------------------------------

# 5. Application Interface

Build a clean, modern, easy-to-use Streamlit application.

## Suggested page layout

### Header

-   Application name: **Smart Document Scanner**
-   Subtitle: **Detect • Correct • Enhance • Analyze • Extract**
-   A short description of what the tool does.

### Sidebar

Include:

-   Upload image.
-   Auto Scan button.
-   Processing options.
-   Brightness slider.
-   Contrast slider.
-   Sharpening slider.
-   Noise-reduction controls.
-   Shadow-removal toggle.
-   OCR language selection, based on available installed language data.

### Main area

Use tabs or clearly separated sections:

1.  **Original Image**
2.  **Document Detection**
3.  **Corrected Scan**
4.  **Enhancement & Filters**
5.  **OCR Text**
6.  **Quality Report**

### Image comparison

Where practical, show original and processed images side by side.

### Intermediate processing results

Provide an expandable section named "Processing Steps" showing relevant
intermediate images, such as:

-   Grayscale image.
-   Blurred image.
-   Canny edges.
-   Detected contour.
-   Perspective-corrected image.
-   Final enhanced image.

These visualizations are important for demonstrating the Digital Image
Processing techniques during the project presentation.

### Downloads

Provide downloads for:

-   Final processed image as PNG or JPG.
-   Grayscale output.
-   Black-and-white output.
-   High-contrast output.
-   Extracted OCR text as TXT.

Only show download buttons for outputs that have actually been
generated.

------------------------------------------------------------------------

# 6. Technical Architecture

Keep the code modular. Do not put the entire application in one Python
file.

Suggested project structure:

``` text
smart-document-scanner/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── scanner/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── detection.py
│   ├── perspective.py
│   ├── deskew.py
│   ├── enhancement.py
│   ├── filters.py
│   ├── noise_removal.py
│   ├── ocr.py
│   ├── quality.py
│   └── pipeline.py
│
├── utils/
│   ├── __init__.py
│   ├── image_io.py
│   ├── geometry.py
│   └── validation.py
│
├── tests/
│   ├── test_geometry.py
│   ├── test_detection.py
│   ├── test_perspective.py
│   ├── test_quality.py
│   └── test_filters.py
│
└── assets/
    └── .gitkeep
```

You may adjust the structure if there is a clear technical reason, but
preserve separation between UI, processing, and utilities.

## Responsibilities of modules

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit UI and user interaction

  `preprocessing.py`                  Grayscale conversion, resizing,
                                      Gaussian blur

  `detection.py`                      Canny edges, contours, document
                                      boundary detection

  `perspective.py`                    Corner ordering and perspective
                                      transformation

  `deskew.py`                         Hough-based angle estimation and
                                      rotation

  `enhancement.py`                    Brightness, contrast, histogram
                                      processing, sharpening

  `filters.py`                        Grayscale, global threshold,
                                      adaptive threshold, high contrast

  `noise_removal.py`                  Median/Gaussian filtering and
                                      shadow reduction

  `ocr.py`                            Tesseract integration and text
                                      extraction

  `quality.py`                        Brightness, contrast, sharpness,
                                      noise and overall score

  `pipeline.py`                       Coordinates the complete automatic
                                      scanning workflow

  `image_io.py`                       Image decoding, encoding, and
                                      format conversion

  `geometry.py`                       Point ordering and geometric
                                      calculations

  `validation.py`                     Input and parameter validation
  -----------------------------------------------------------------------

------------------------------------------------------------------------



Use clear, typed function signatures where practical. The following are
suggested interfaces; improve them if needed while keeping
responsibilities clear.

``` python
# preprocessing.py
def to_grayscale(image):
    ...

def reduce_initial_noise(image, kernel_size=5):
    ...

def resize_for_detection(image, max_dimension=1200):
    ...


# detection.py
def detect_document_corners(image):
    """
    Return ordered corner coordinates and detection metadata.
    If no reliable document is detected, return a clear failure result.
    """
    ...


# perspective.py
def order_points(points):
    ...

def correct_perspective(image, corners):
    ...


# deskew.py
def estimate_skew_angle(image):
    ...

def correct_skew(image, max_correction_angle=15):
    ...


# enhancement.py
def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    ...

def enhance_contrast(image):
    ...

def sharpen_image(image, strength=1.0):
    ...


# filters.py
def convert_to_grayscale(image):
    ...

def global_threshold(image, threshold_value=127):
    ...

def adaptive_threshold(image, block_size=21, c_value=10):
    ...

def create_high_contrast(image):
    ...


# noise_removal.py
def median_denoise(image, kernel_size=3):
    ...

def gaussian_denoise(image, kernel_size=5):
    ...

def reduce_shadows(image):
    ...


# ocr.py
def extract_text(image, language="eng"):
    ...

def extract_text_with_confidence(image, language="eng"):
    ...


# quality.py
def calculate_brightness(image):
    ...

def calculate_contrast(image):
    ...

def calculate_sharpness(image):
    ...

def estimate_noise(image):
    ...

def calculate_quality_score(image, detection_success=True):
    ...


# pipeline.py
def scan_document(image, options=None):
    """
    Run the end-to-end pipeline and return a structured result
    containing the processed image, intermediate results, detection
    information, generated filter outputs, and status messages.
    """
    ...
```

Use dataclasses or clearly documented dictionaries for structured
results. Avoid returning inconsistent data types from different
branches.

-----------------------------------------------------------------------

# 8. End-to-End Processing Pipeline

Implement the automatic pipeline in this order:

1.  Validate and decode the input image.
2.  Preserve an untouched copy of the original.
3.  Create a working image for preprocessing and detection.
4.  Convert to grayscale.
5.  Apply Gaussian blur.
6.  Detect edges using Canny.
7.  Find contours.
8.  Identify a plausible four-corner document boundary.
9.  If reliable corners are found, apply perspective correction.
10. If no reliable corners are found, retain the original image and
    report that perspective correction was skipped.
11. Estimate and correct small skew.
12. Apply configurable image enhancement.
13. Apply noise reduction.
14. Optionally apply shadow reduction.
15. Generate grayscale output.
16. Generate global/adaptive black-and-white output.
17. Generate high-contrast output.
18. Calculate quality metrics on the final processed scan.
19. Run OCR on the processed scan.
20. Return all results to the UI.

The order may be adjusted when technically justified. In particular,
avoid applying a destructive operation twice or applying an operation
that makes later detection less reliable.

The pipeline must preserve intermediate outputs for demonstration and
debugging.

------------------------------------------------------------------------

# 9. Error Handling and Edge Cases

The application must handle:

-   No image uploaded.
-   Unsupported file type.
-   Corrupted image.
-   Image too small for reliable processing.
-   Very large image.
-   No document contour detected.
-   More than one plausible rectangular contour.
-   Invalid or crossed corner coordinates.
-   Perspective transformation failure.
-   Skew angle estimation failure.
-   Missing Tesseract executable.
-   Missing Tesseract language data.
-   OCR returning empty text.
-   Invalid slider or processing parameters.
-   Processing exceptions.

Do not crash the entire application because one optional feature fails.

For example, if OCR is unavailable, the user should still be able to
scan, enhance, analyze, and download the document. Display an actionable
OCR setup message.

------------------------------------------------------------------------

# 10. Testing Requirements

Write unit tests for the important image-processing functions.

At minimum, test:

## Geometry

-   Four points are ordered correctly.
-   Perspective output dimensions are valid.
-   Invalid corner input is rejected.

## Detection

-   A synthetic image containing a clear rectangle can be detected.
-   A blank image does not produce a false successful detection.
-   Detection failure returns a predictable result.

## Filters

-   Grayscale conversion returns a single-channel image.
-   Thresholding produces a binary image.
-   Output dimensions are preserved where expected.

## Quality analysis

-   Brightness metric returns a valid numerical value.
-   Contrast metric returns a valid numerical value.
-   Sharpness metric returns a valid numerical value.
-   Overall score is between 0 and 100.
-   The score is calculated from image metrics rather than a hard-coded
    value.
-   Invalid images are handled safely.

## Pipeline

-   The pipeline returns a structured result.
-   Detection failure does not crash the pipeline.
-   Original input is not modified.

Use synthetic test images where possible so tests do not depend on
private documents or external datasets.

------------------------------------------------------------------------

# 11. Dependencies and Environment

Create a `requirements.txt` containing the required Python packages.

Suggested dependencies:

``` text
streamlit
opencv-python
numpy
Pillow
pytesseract
pytest
```

Add other dependencies only when genuinely needed.

## Windows setup instructions

Document these steps in the README:

1.  Install Python.
2.  Create a virtual environment:

``` powershell
python -m venv .venv
```

3.  Activate it:

``` powershell
.venv\Scripts\Activate.ps1
```

4.  Install dependencies:

``` powershell
pip install -r requirements.txt
```

5.  Install the Tesseract OCR application separately on Windows.
6.  Ensure the Tesseract executable is available in PATH, or let the
    user configure its path in the application.
7.  Start the app:

``` powershell
streamlit run app.py
```

8.  Run tests:

``` powershell
pytest -q
```

The Python package `pytesseract` is only a wrapper. It does not install
the Tesseract OCR executable or its language files. Explain this
distinction in the README.

------------------------------------------------------------------------

# 12. Non-Functional Requirements

-   The application should run on a normal Windows laptop without a GPU.
-   The interface should be understandable to a first-time user.
-   Processing functions should be modular and reusable.
-   The app should not modify the original uploaded image.
-   The app should not transmit documents to external services.
-   Errors should be displayed in understandable language.
-   The application should avoid unnecessary processing delays.
-   The code should be readable and maintainable.
-   All major processing steps should be demonstrable during an academic
    presentation.

------------------------------------------------------------------------

# 13. Expected Deliverables

The completed project should contain:

1.  A working Streamlit document-scanning application.
2.  Automatic document boundary detection.
3.  Perspective correction.
4.  Automatic deskewing.
5.  Brightness, contrast, and sharpness enhancement.
6.  Noise and shadow reduction.
7.  Original, grayscale, black-and-white, and high-contrast outputs.
8.  OCR-based printed text extraction.
9.  Scan-quality analysis with a score and individual metrics.
10. Download options.
11. Intermediate processing visualizations.
12. Modular, documented source code.
13. Unit tests.
14. A README with installation and usage instructions.

------------------------------------------------------------------------

# 14. Development Phases

Build in phases. Do not attempt to implement everything in one large,
untested change.

## Phase 1 --- Project foundation

-   Inspect repository.
-   Set up project structure.
-   Create requirements file.
-   Build basic Streamlit upload interface.
-   Display original image.
-   Validate image input.

**Completion condition:** User can upload and preview a valid image.

## Phase 2 --- Preprocessing and document detection

-   Grayscale conversion.
-   Gaussian blur.
-   Canny edge detection.
-   Contour detection.
-   Four-corner approximation.
-   Boundary visualization.

**Completion condition:** The application can detect and display a clear
document boundary in suitable images.

## Phase 3 --- Perspective correction and deskewing

-   Point ordering.
-   Homography calculation.
-   Perspective warp.
-   Hough line-based angle estimation.
-   Image rotation.

**Completion condition:** A document photographed at a moderate angle
can be transformed into a rectangular view, and small rotations can be
corrected.

## Phase 4 --- Enhancement and artifact removal

-   Brightness and contrast.
-   Histogram equalization / CLAHE.
-   Sharpening.
-   Median and Gaussian filtering.
-   Shadow reduction.

**Completion condition:** User can see and control enhancement
operations without losing the original image.

## Phase 5 --- Document filters

-   Grayscale mode.
-   Global thresholding.
-   Adaptive thresholding.
-   High-contrast mode.

**Completion condition:** All required scanner-style outputs can be
generated and downloaded.

## Phase 6 --- OCR integration

-   Tesseract detection and configuration.
-   Text extraction.
-   Language handling.
-   TXT download.

**Completion condition:** Suitable printed documents produce visible OCR
text, and missing OCR dependencies produce a helpful message.

## Phase 7 --- Scan Quality Analyzer

-   Brightness metric.
-   Contrast metric.
-   Sharpness metric.
-   Noise estimate.
-   Boundary validation.
-   Normalization and weighted scoring.
-   Quality labels and report.

**Completion condition:** The score is derived from actual image
measurements and all component metrics are visible.

## Phase 8 --- Integration and final testing

-   Integrate the full pipeline.
-   Add intermediate visualizations.
-   Improve interface layout.
-   Add error handling.
-   Run tests.
-   Test with multiple document photographs.
-   Update README.

**Completion condition:** The application can complete the end-to-end
workflow and export its outputs.

------------------------------------------------------------------------

# 15. Acceptance Criteria

The project is considered complete only when:

-   [ ] A user can upload a JPG, JPEG, or PNG document photograph.
-   [ ] The original image is displayed and preserved.
-   [ ] The system can detect a clear document boundary in suitable
    images.
-   [ ] Detected document corners can be visualized.
-   [ ] Perspective correction produces a rectangular document output.
-   [ ] Detection failure is handled without crashing.
-   [ ] Small document skew can be estimated and corrected.
-   [ ] Brightness, contrast, and sharpness can be adjusted.
-   [ ] Noise reduction is implemented.
-   [ ] Shadow reduction is implemented with an option to disable it.
-   [ ] Grayscale output is generated.
-   [ ] Black-and-white output is generated.
-   [ ] High-contrast output is generated.
-   [ ] OCR extracts text from suitable printed documents when Tesseract
    is installed.
-   [ ] OCR text can be downloaded.
-   [ ] Brightness, contrast, sharpness, and noise metrics are
    calculated.
-   [ ] The overall quality score is calculated from real metrics.
-   [ ] The quality score is bounded between 0 and 100.
-   [ ] The user can download processed images.
-   [ ] Intermediate image-processing stages can be inspected.
-   [ ] The README explains installation and execution.
-   [ ] Automated tests pass.

------------------------------------------------------------------------

# 16. Limitations to Acknowledge

The application is designed primarily for a single document photographed
against a reasonably distinguishable background.

Expected limitations include:

-   Very low-light or heavily blurred photographs may reduce document
    detection and OCR accuracy.
-   Documents with backgrounds having similar edges or colors may be
    harder to isolate.
-   Severe perspective distortion may make reliable corner detection
    difficult.
-   OCR accuracy depends on text size, font, language, image quality,
    and preprocessing.
-   The quality score is an image-based estimate and must not be
    interpreted as a universal measure of document quality.
-   Handwriting recognition is not part of the required initial scope.
-   Multi-page PDF creation, automatic page ordering, document
    classification, and cloud storage are future extensions, not core
    requirements.

Do not claim that these limitations have been solved unless a tested
implementation genuinely addresses them.

------------------------------------------------------------------------

# 17. Future Enhancements (Not Required for Initial Version)

Keep the architecture extensible for possible future additions:

-   Multi-page PDF creation.
-   Automatic page ordering.
-   Handwriting recognition.
-   Document classification.
-   Batch document scanning.
-   Cloud storage.
-   PDF export.
-   Manual corner adjustment through an interactive interface.
-   More advanced illumination correction.
-   OCR confidence visualization.

Do not prioritize these over the core requirements.

------------------------------------------------------------------------

# 18. Final Instructions to Claude Code

Start by inspecting the repository and reporting:

1.  What files already exist.
2.  Which parts can be reused.
3.  What needs to be created.
4.  Any environment or dependency issues you identify.

Then implement the project phase by phase.

After each phase:

-   Run relevant tests.
-   Fix errors.
-   Summarize what works.
-   Identify any known limitations.
-   Continue to the next phase.

At the end, provide:

-   The final project structure.
-   Exact installation commands for Windows PowerShell.
-   Exact command to run the application.
-   Exact command to run tests.
-   Tesseract installation/configuration instructions.
-   A concise list of implemented features.
-   Any limitations or features that remain incomplete.

**Priority:** A reliable, demonstrable end-to-end scanner with genuine
image-processing operations is more important than a visually elaborate
interface. Every displayed metric and result must come from actual
processing, not mock data.
