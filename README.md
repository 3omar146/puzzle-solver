# Puzzle Solver – Image-Based Jigsaw Reconstruction

This project implements an **end-to-end computer vision pipeline** for solving grid-based jigsaw puzzles (2×2, 4×4, 8×8) from a single image.
It processes an input image, extracts puzzle pieces, analyzes their edges, and reconstructs the original puzzle using **color-based edge matching**.

The system is modular, allowing individual pipeline stages to be inspected, replaced, or run in-memory for debugging.

---

## High-Level Pipeline Overview

1. **Image Enhancement**
2. **Grid Segmentation**
3. **Piece Preprocessing**

   * Thresholding
   * Edge extraction (diagnostic)
4. **Edge Feature Extraction (Color-Based)**
5. **Puzzle Reconstruction**
6. **Assembly**

Each stage is implemented as a separate module.

---

## 1. Image Enhancement

**Goal:** Improve contrast and local detail to make grid boundaries and piece edges more reliable.

### Techniques Used

* **Grayscale conversion**
* **Bilateral filtering**

  * Reduces noise while preserving edges
* **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

  * Enhances local contrast
* **Unsharp masking**

  * Sharpens edges by amplifying high-frequency details

### Output

* Enhanced grayscale image (used for segmentation and diagnostics)
* Original color image is preserved for reconstruction

---

## 2. Grid Segmentation

**Goal:** Split the image into equal-sized puzzle pieces based on a known grid size.

### Method

* Image is divided into an `N × N` grid
* Each tile is cropped directly using pixel coordinates
* Bounding boxes are optionally drawn for visualization

### Notes

* Grid size can be:

  * Fixed (e.g. inferred from dataset name)
  * Automatically detected (optional module, not required)

---

## 3. Piece Preprocessing (Diagnostics)

These steps are **not used directly for reconstruction**, but help visualize and debug piece quality.

### Thresholding

* **Adaptive Gaussian Thresholding**

  * Handles uneven lighting
  * Produces a clean binary mask of the piece

### Edge Detection

* **Canny Edge Detector**

  * Automatically selects thresholds using image statistics
  * Produces thin, clean edges

These outputs are useful for inspection but **not used in matching**.

---

## 4. Edge Feature Extraction (Color-Based)

**This is the core matching signal of the solver.**

### Edge Slicing

For each color puzzle piece:

* Extract thin strips from:

  * Top
  * Right
  * Bottom
  * Left

### Color Space

* Convert edge strips from **BGR → HSV**
* Use **Hue + Saturation** channels (more stable than RGB)

### Background Suppression

* Pixels close to the mean brightness are treated as background
* Reduces influence of flat or empty regions

### Feature Representation

* **Normalized 2D color histogram** (H × S)
* Histogram normalization ensures scale invariance

---

## 5. Edge Matching Algorithm

### Distance Metric

* **Bhattacharyya distance** between histograms
* Lower distance = better match

### Orientation Handling

* Candidate edges are compared **with and without flipping**
* Ensures correct orientation is found automatically

---

## 6. Puzzle Reconstruction

### 2×2 Solver

* Brute-force over all permutations (4! = 24)
* Score each configuration by summing adjacent edge distances
* Select configuration with lowest total cost

### NxN Solver (4×4, 8×8)

* Greedy placement:

  * Fill grid left-to-right, top-to-bottom
  * At each position, choose piece minimizing mismatch with already-placed neighbors
* Uses local consistency rather than global optimization

---

## 7. Assembly

* Selected pieces are stitched together into a final image
* Assembly preserves original resolution and color

---

## Design Principles

* **Modularity:** Each stage can be run independently
* **Separation of concerns:**
* 
  * Grayscale → processing
  * Color → reconstruction
* **Deterministic pipeline:** No learning, no randomness
* **Debug-friendly:** Intermediate results are easily visualized

---

## Key Assumptions & Limitations

* Puzzle pieces form a **perfect grid**
* No rotation of pieces (orientation is fixed)
* Reconstruction is **color-based**, not shape-based
* NxN solver is greedy, not globally optimal
