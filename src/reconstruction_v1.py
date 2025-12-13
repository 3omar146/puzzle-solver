import cv2
import numpy as np
from itertools import permutations
from pathlib import Path

from src.paths import EDGE_PIECES_DIR, COLORED_PIECES_DIR

# EDGE SLICING
def slice_edges_color(img):
    border = 5
    return (
        img[:border, :, :],    # top
        img[:, -border:, :],   # right
        img[-border:, :, :],   # bottom
        img[:, :border, :],    # left
    )


# COLOR HISTOGRAM EDGE DISTANCE
def _background_mask(hsv_edge, v_thresh=10):
    """
    Suppress background pixels using V-channel deviation.
    Pixels whose brightness differs from the mean by more than v_thresh
    are considered foreground.
    """
    v = hsv_edge[:, :, 2].astype(np.int16)
    mean_v = np.mean(v)
    return (np.abs(v - mean_v) > v_thresh).astype(np.uint8)


def _edge_histogram(hsv_edge, mask):
    """
    Compute a normalized HS histogram for an edge strip.
    Uses H+S only (more stable than full HSV).
    """
    if mask is not None and int(mask.sum()) == 0:
        mask = None

    hist = cv2.calcHist(
        [hsv_edge],
        [0, 1], # H, S channels
        mask,
        [16, 16], # bins
        [0, 180, 0, 256]
    )

    # Normalize
    s = float(hist.sum())
    if s <= 0:
        hist = np.ones_like(hist, dtype=np.float32)
        s = float(hist.sum())

    return (hist / s).astype(np.float32)


def edge_color_distance(a_bgr, b_bgr, orientation="vertical"):
    """
    Compute color distance between two edge strips.
    """

    hsv_a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)

    # Choose correct flip axis
    flip_axis = 0 if orientation == "vertical" else 1
    hsv_bf = cv2.flip(hsv_b, flip_axis)

    # Background suppression
    mask_a  = _background_mask(hsv_a)
    mask_b  = _background_mask(hsv_b)
    mask_bf = _background_mask(hsv_bf)

    # Histograms
    h1  = _edge_histogram(hsv_a,  mask_a)
    h2  = _edge_histogram(hsv_b,  mask_b)
    h2f = _edge_histogram(hsv_bf, mask_bf)

    # Bhattacharyya distance (lower is better)
    d1 = cv2.compareHist(h1, h2,  cv2.HISTCMP_BHATTACHARYYA)
    d2 = cv2.compareHist(h1, h2f, cv2.HISTCMP_BHATTACHARYYA)

    d = min(d1, d2)

    if not np.isfinite(d):
        return 1e9

    return d


def start_reconstruction_v1(color_pieces, grid_size):
    n = grid_size * grid_size
    if len(color_pieces) != n:
        raise ValueError(f"Expected {n} pieces, got {len(color_pieces)}")

    # Build piece dicts
    pieces = []
    for img in color_pieces:
        top, right, bottom, left = slice_edges_color(img)
        pieces.append({
            "img": img,
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
        })

    # Solve
    if grid_size == 2:
        tl, tr, bl, br = solve_2x2(pieces) # type: ignore
        grid = [[tl, tr], [bl, br]]
    else:
        grid = solve_NxN(pieces, grid_size)

    # Assemble
    h, w = color_pieces[0].shape[:2]
    canvas = np.zeros((grid_size*h, grid_size*w, 3), np.uint8)

    for r in range(grid_size):
        for c in range(grid_size):
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = grid[r][c]["img"]

    return canvas


# SOLVERS
def solve_2x2(pieces):
    best_combo = None
    lowest = float("inf")

    for perm in permutations(pieces):
        tl, tr, bl, br = perm
        val = (
            edge_color_distance(tl["right"], tr["left"], "vertical") +
            edge_color_distance(tl["bottom"], bl["top"], "horizontal") +
            edge_color_distance(tr["bottom"], br["top"], "horizontal") +
            edge_color_distance(bl["right"], br["left"], "vertical")
        )
        if val < lowest:
            lowest = val
            best_combo = perm

    return best_combo

def solve_NxN(pieces, size):
    board = [[None] * size for _ in range(size)]
    pool = list(pieces)

    for r in range(size):
        for c in range(size):
            best = None
            best_cost = float("inf")
            best_idx = -1

            for i, piece in enumerate(pool):
                total = 0.0

                if c > 0:
                    left = board[r][c - 1]
                    total += edge_color_distance(
                        left["right"], piece["left"], "vertical"
                    )

                if r > 0:
                    top = board[r - 1][c]
                    total += edge_color_distance(
                        top["bottom"], piece["top"], "horizontal"
                    )

                if not np.isfinite(total):
                    total = 1e9

                if total < best_cost:
                    best_cost = total
                    best = piece
                    best_idx = i

            if best is None:
                best_idx = 0
                best = pool[0]

            board[r][c] = best
            pool.pop(best_idx)

    return board


# ASSEMBLY
def assemble_2x2(a, b, c, d):
    imgs = [cv2.imread(x["src"]) for x in (a, b, c, d)]
    if any(im is None for im in imgs):
        missing = [i for i, im in enumerate(imgs) if im is None]
        raise FileNotFoundError(f"assemble_2x2: missing images at indices {missing}")

    h, w = imgs[0].shape[:2] # type: ignore
    canvas = np.zeros((2*h, 2*w, 3), np.uint8)
    canvas[:h, :w] = imgs[0]
    canvas[:h, w:] = imgs[1]
    canvas[h:, :w] = imgs[2]
    canvas[h:, w:] = imgs[3]
    return canvas

def assemble_NxN(grid):
    n = len(grid)
    base = cv2.imread(grid[0][0]["src"])
    if base is None:
        raise FileNotFoundError(f"assemble_NxN: cannot read {grid[0][0]['src']}")

    h, w = base.shape[:2]
    canvas = np.zeros((n*h, n*w, 3), np.uint8)

    for r in range(n):
        for c in range(n):
            img = cv2.imread(grid[r][c]["src"])
            if img is None:
                raise FileNotFoundError(f"assemble_NxN: cannot read {grid[r][c]['src']}")
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img

    return canvas
