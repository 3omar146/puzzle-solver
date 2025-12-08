import cv2
import numpy as np

def pixel_accuracy(pred_path, truth_path, grid_size):
    pred = cv2.imread(pred_path)
    truth = cv2.imread(truth_path)

    if pred is None or truth is None:
        return None, None, None

    H, W = pred.shape[:2]
    truth = cv2.resize(truth, (W, H))

    # Seam mask: ignore tile borders
    mask = np.ones((H, W), dtype=np.uint8)

    step_h = H // grid_size
    step_w = W // grid_size

    # ignore 2px border around seams
    seam_thickness = 2

    for i in range(1, grid_size):
        mask[i * step_h - seam_thickness : i * step_h + seam_thickness, :] = 0
        mask[:, i * step_w - seam_thickness : i * step_w + seam_thickness] = 0

    # Blur-based pixel comparison
    pred_blur = cv2.GaussianBlur(pred,  (3,3), 0)
    truth_blur = cv2.GaussianBlur(truth, (3,3), 0)

    diff = np.abs(pred_blur.astype(np.int16) - truth_blur.astype(np.int16))
    correct_pixels = np.sum(np.all(diff < 20, axis=2) & (mask == 1))
    total_pixels = np.sum(mask == 1)

    acc = correct_pixels / total_pixels * 100.0
    return acc, correct_pixels, total_pixels
