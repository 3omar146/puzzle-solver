import cv2
import numpy as np

def binary_edges(img):
    """
    Extracts ONLY the inner boundary from a binary puzzle mask.
    Produces thin, clean edges.
    """
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(img, kernel)

    edges = cv2.subtract(img, eroded)
    return edges

def canny_edges(img):
    """
    Extracts edges using Canny with automatic threshold selection.
    """
    median = np.median(img)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    edges = cv2.Canny(img, lower, upper)

    return edges

def laplacian_edges(img):
    """
    Extract edges using the Laplacian operator.
    """

    # 1. Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Light smoothing to reduce Laplacian noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Apply Laplacian
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # 4. Take absolute values to avoid negative edges
    lap_abs = np.absolute(lap)

    # 5. Normalize to 0–255 uint8
    lap_norm = np.uint8(255 * lap_abs / lap_abs.max())

    return lap_norm