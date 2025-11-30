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
