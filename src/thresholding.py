import cv2
import numpy as np

def threshold_image(enhanced_gray_image):
    """
    Takes an enhanced grayscale image and applies adaptive thresholding to produce
    a clean binary mask where puzzle pieces appear as white regions.

    Parameters
    ----------
    enhanced_gray_image : np.ndarray
        Grayscale image after Step 1 enhancement.

    Returns
    -------
    binary_mask : np.ndarray
        Thresholded binary image suitable for contour extraction (Step 3).
    """
    
    # Heavy blur to remove texture details
    blurred = cv2.GaussianBlur(enhanced_gray_image, (21, 21), 0)

    # Otsu thresholding works better on heavily smoothed images
    _, binary_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary_mask