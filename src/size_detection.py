import cv2
import numpy as np


def detect_grid_size(img, img_clahe):
    grid_raw, conf_raw = compute_grid(img)
    grid_clahe, conf_clahe = compute_grid(img_clahe)

    # choose whichever produced stronger grid evidence
    return grid_raw if conf_raw >= conf_clahe else grid_clahe


def compute_grid(img):
    h, w = img.shape

    # Enhanced grayscale using local variance
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    boosted = cv2.addWeighted(img, 0.5, lap, 0.5, 0)

    # Sobel edges
    sx = cv2.Sobel(boosted, cv2.CV_64F, 1, 0, 3) # type: ignore
    sy = cv2.Sobel(boosted, cv2.CV_64F, 0, 1, 3) # type: ignore
    ax = cv2.convertScaleAbs(sx)
    ay = cv2.convertScaleAbs(sy)

    # Edge projections
    vx = np.sum(ax, axis=0)
    hy = np.sum(ay, axis=1)

    # Smooth
    k = max(3, int(min(h, w) * 0.005))
    smooth = np.ones(k) / k
    vx = np.convolve(vx, smooth, mode='same')
    hy = np.convolve(hy, smooth, mode='same')

    # Normalize
    v_norm = vx / (np.median(vx) + 1e-5)
    h_norm = hy / (np.median(hy) + 1e-5)

    # Grid scoring helper
    def grid_score(proj, N):
        if N == 8:
            ratios = [1/8, 3/8, 5/8, 7/8]
            thr = 1.5
        elif N == 4:
            ratios = [1/4, 3/4]
            thr = 2.0
        else:
            ratios = [1/2]
            thr = 2.0

        L = len(proj)
        win = int(L * 0.02)

        peaks = []
        for r in ratios:
            c = int(L * r)
            s = max(0, c - win)
            e = min(L, c + win)
            peaks.append(np.max(proj[s:e]))

        return sum(p > thr for p in peaks) / len(peaks)

    # Grid strengths
    pv8 = grid_score(v_norm, 8)
    ph8 = grid_score(h_norm, 8)
    pv4 = grid_score(v_norm, 4)
    ph4 = grid_score(h_norm, 4)

    # Decisions
    is8 = (pv8 + ph8) >= 1.5
    is4 = (pv4 + ph4) >= 1.0

    if is8: return 8, pv8 + ph8
    if is4: return 4, pv4 + ph4

    return 2, 0.0
