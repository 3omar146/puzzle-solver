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
    Input can be grayscale or BGR.
    Output: clean 1-pixel edges (white on black).
    """


    # 3. Auto thresholding for Canny based on median intensity
    median = np.median(img)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    # 4. Canny detection
    edges = cv2.Canny(img, lower, upper)

     # 3. Morphological cleanup
    #kernel = np.ones((2,2), np.uint8)
    #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges