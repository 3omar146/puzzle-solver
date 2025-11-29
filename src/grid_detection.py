import cv2
import numpy as np


def detect_grid_size(img):
    h, w = img.shape

    # Local variance map (helps distinguish 2×2)
    var_map = cv2.Laplacian(img, cv2.CV_64F)
    var_map = cv2.convertScaleAbs(var_map)
    gray_boost = cv2.addWeighted(img, 0.5, var_map, 0.5, 0)

    # Sobel edges
    sx = cv2.Sobel(gray_boost, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_boost, cv2.CV_64F, 0, 1, ksize=3)
    ax = cv2.convertScaleAbs(sx)
    ay = cv2.convertScaleAbs(sy)

    # Projections
    vx = np.sum(ax, axis=0)
    hy = np.sum(ay, axis=1)

    # Smooth projections
    k = max(3, int(min(h, w) * 0.005))
    vx = np.convolve(vx, np.ones(k)/k, mode='same')
    hy = np.convolve(hy, np.ones(k)/k, mode='same')

    # Normalize
    v_norm = vx / (np.median(vx) + 1e-5)
    h_norm = hy / (np.median(hy) + 1e-5)

    # Grid score helper
    def score(proj, N):
        L = len(proj)
        if N == 8:
            ratios = [1/8, 3/8, 5/8, 7/8]
        elif N == 4:
            ratios = [1/4, 3/4]
        else:
            ratios = [1/2]

        win = int(L * 0.02)
        peaks = []

        for r in ratios:
            c = int(L * r)
            s = max(0, c - win)
            e = min(L, c + win)
            peaks.append(np.max(proj[s:e]))

        thr = 1.5 if N == 8 else 2.0
        pass_rate = sum(p > thr for p in peaks) / len(peaks)
        return pass_rate, np.mean(peaks)

    # Tile variance
    def tile_variance_score(gray, k):
        th = h // k
        tw = w // k
        vals = []
        for r in range(k):
            for c in range(k):
                tile = gray[r*th:(r+1)*th, c*tw:(c+1)*tw]
                vals.append(np.var(tile))
        return np.mean(vals)

    var2 = tile_variance_score(img, 2)
    var4 = tile_variance_score(img, 4)
    var8 = tile_variance_score(img, 8)

    # Grid line scores
    pv8, _ = score(v_norm, 8)
    ph8, _ = score(h_norm, 8)
    pv4, _ = score(v_norm, 4)
    ph4, _ = score(h_norm, 4)

    is8 = (pv8 >= 0.60 and ph8 >= 0.60)
    is4 = (pv4 + ph4) >= 1.00
    is2 = (var2 >= var4 * 1.00) and (var2 >= var8 * 1.125)

    if is8: return 8
    if is4: return 4
    if is2: return 2

    is8 = var8 < var4 * 0.985
    is4 = (var4 < var2 * 0.98)
    is2 = var2 > var4 * 1.015

    if ph4 == 0 and pv4 == 0 and not is2: 
        if is8: return 8
        if is4: return 4

    if is2: return 2

    print("PV8: {:.2f}, PH8: {:.2f}, PV4: {:.2f}, PH4: {:.2f}".format(pv8, ph8, pv4, ph4))
    print("Var2: {:.2f}, Var4: {:.2f}, Var8: {:.2f}".format(var2, var4, var8))