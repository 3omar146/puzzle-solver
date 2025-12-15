# Puzzle Solver – Image-Based Jigsaw Reconstruction

This project implements a **classical computer vision pipeline** for reconstructing grid-based jigsaw puzzles (2×2, 4×4, 8×8) from a single input image.

The system preprocesses an image containing a complete puzzle, segments it into pieces, analyzes piece edges, and reconstructs the original puzzle using **purely deterministic, non-ML techniques**.

Two reconstruction algorithms are provided:

* **Reconstruction v1:** Greedy, local edge matching
* **Reconstruction v2:** Global seam scoring with cluster-based assembly

Both solvers share the same preprocessing pipeline and can be compared directly.

---

## High-Level Pipeline Overview

1. **Image Enhancement**
2. **Grid Segmentation**
3. **Piece Preprocessing (Diagnostics)**
4. **Edge / Seam Feature Extraction**
5. **Puzzle Reconstruction (v1 or v2)**
6. **Final Assembly**

Each stage is implemented as an independent module to facilitate debugging and experimentation.

---

## 1. Image Enhancement

**Goal:** Improve local contrast and edge clarity to ensure reliable segmentation and seam analysis.

### Techniques Used

* **Grayscale conversion**
* **Bilateral filtering**

  * Reduces noise while preserving edges
* **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

  * Enhances local contrast under uneven lighting
* **Unsharp masking**

  * Sharpens high-frequency details and edges

### Output

* Enhanced grayscale image (used for segmentation and diagnostics)
* Original color image is preserved for reconstruction

---

## 2. Grid Segmentation

**Goal:** Split the puzzle image into equal-sized pieces based on a known grid size.

### Method

* The image is divided into an `N × N` grid
* Each tile is cropped using pixel coordinates
* Optional bounding-box visualization is provided for debugging

### Notes

* Grid size can be:

  * Fixed (derived from dataset structure), or
  * Automatically detected (optional module)

---

## 3. Piece Preprocessing (Diagnostics Only)

These steps are used for **inspection and debugging only** and are **not part of the reconstruction algorithms**.

### Thresholding

* **Adaptive Gaussian thresholding**
* Produces clean binary masks of individual pieces

### Edge Detection

* **Canny edge detector**
* Automatic threshold selection
* Produces thin, well-localized edges

These outputs help evaluate segmentation quality but are not used for matching.

---

## 4. Seam Feature Extraction (Color-Based)

Reconstruction is driven by **color continuity across piece boundaries**.

### Color Space

* Puzzle pieces are converted from **BGR → LAB**
* LAB is perceptually uniform and better suited for seam comparison

### Seam Definition

For each piece, four seams are considered:

* Left
* Right
* Top
* Bottom

Seams are extracted as thin pixel bands along the piece border.

---

## 5. Reconstruction v1 – Greedy Local Assembly

### Overview

Reconstruction v1 uses a **greedy placement strategy** based on local seam similarity.

### Seam Matching

* Adjacent seams are compared using mean absolute LAB color difference
* Lower difference indicates a better match

### Solvers

#### 2×2 Puzzle

* Brute-force evaluation of all permutations (4! = 24)
* Total seam cost is computed for each configuration
* Configuration with minimum cost is selected

#### NxN Puzzle (4×4, 8×8)

* Pieces are placed sequentially:

  * Left-to-right, top-to-bottom
* At each grid cell:

  * All remaining pieces are evaluated
  * The piece minimizing mismatch with already placed neighbors is selected

### Properties

* Simple and fast
* Susceptible to early placement errors
* No global consistency enforcement

---

## 6. Reconstruction v2 – Global Seam Clustering

Reconstruction v2 is a **global optimization-inspired solver** designed to overcome the limitations of greedy placement.

### Core Idea

Instead of placing pieces sequentially, v2:

* Scores **all possible pairwise adjacencies**
* Ranks candidate matches using multiple confidence signals
* Merges pieces into clusters using a **Kruskal-style strategy**
* Produces a globally consistent layout before final assembly

---

### 6.1 Seam Cost and Reliability

For each ordered pair of pieces:

* **Horizontal adjacency:** right → left
* **Vertical adjacency:** bottom → top

#### Seam Cost

* Mean absolute LAB difference across the seam

#### Seam Variance (Reliability Measure)

* L-channel variance along the seam
* Interpretation:

  * Low variance → flat/background edge (unreliable)
  * High variance → textured edge (reliable)

Self-matching is explicitly disallowed.

---

### 6.2 Candidate Edge Scoring

Each candidate adjacency is scored using three signals:

1. **Ratio Test (Uniqueness)**
   Measures how distinctive the best match is compared to the second-best:

   ```
   ratio = best_cost / second_best_cost
   ```

2. **Mutual-Best Agreement (Global Consistency)**
   A strong bonus is applied if:

   * Piece A selects B as its best neighbor
   * Piece B independently selects A as its best neighbor

   This is the strongest confidence signal.

3. **Variance-Based Confidence Penalty (Local Reliability)**
   Flat seams are penalized smoothly:

   ```
   penalty ∝ exp(-variance / scale)
   ```

   * The decay scale is **adaptively estimated from the dataset**
   * Textured seams rapidly lose penalty

Lower total score indicates a better match.

---

### 6.3 Global Cluster Merging

* Each piece starts as a singleton cluster
* Candidate edges are processed in ascending score order
* Clusters are merged if:

  * The adjacency is geometrically consistent
  * No overlap occurs
  * The cluster fits within the grid bounds

This process is conceptually similar to **Kruskal’s minimum spanning tree algorithm** with spatial constraints.

---

### 6.4 Normalization and Completion

After merging:

* The largest valid cluster is selected
* Coordinates are normalized to a top-left origin
* Any missing pieces are placed deterministically into remaining slots (fallback)

---

## 7. Final Assembly

* Pieces are stitched according to the final layout
* Original resolution and color are preserved
* No blending or interpolation is applied

---

## Comparison: Reconstruction v1 vs v2

| Aspect             | v1 (Greedy)          | v2 (Global)       |
| ------------------ | -------------------- | ----------------- |
| Strategy           | Sequential placement | Global clustering |
| Optimization       | Local                | Global            |
| Mutual Consistency | ❌                    | ✅                 |
| Variance Awareness | ❌                    | ✅                 |
| Error Propagation  | High                 | Low               |
| Suitable for 8×8   | ❌                    | ✅                 |

---

## Design Principles

* **Pure classical computer vision**
* **No machine learning**
* **Deterministic and reproducible**
* **Modular pipeline**
* **Debug-friendly intermediate outputs**

---

## Assumptions and Limitations

* Puzzle pieces form a **perfect grid**
* Piece orientation is fixed (no rotation)
* Matching is **color-based**, not shape-based
* Completion of missing slots in v2 is heuristic
