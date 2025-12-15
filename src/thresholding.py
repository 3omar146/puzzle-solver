import cv2
import numpy as np


def threshold_adaptive(img):
    """
    Generate a clean binary mask using adaptive thresholding only.
    """
    
    # 1. Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Adaptive Gaussian thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,  # block size
        5    # C constant
    )

    # 3. Morphological cleanup
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def threshold_otsu(img):
    """
    Global Otsu threshold for cleaner, more stable masks.
    Returns: white piece (255) on black background (0).
    """

    # 1. Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Otsu thresholding
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary