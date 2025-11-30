import cv2
import numpy as np

def threshold(img):
    """
    Generate a clean binary mask using adaptive thresholding only.
    Returns: white piece (255) on black background (0).
    """

    # 1. Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2. Adaptive threshold (Gaussian)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # local Gaussian weighting
        cv2.THRESH_BINARY_INV,           # piece = white, background = black
        25,                              # block size (must be odd)
    5                               # subtractive constant (tunes sensitivity)
    )

    # 3. Morphological cleanup (connect borders, remove noise)
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


    
    return binary
