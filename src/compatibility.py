import cv2
import numpy as np

def extract_edge_strip(img, strip):
    h, w = img.shape[:2]
    t = img[0:strip,:,:]
    b = img[h-strip:h,:,:]
    l = img[:,0:strip,:]
    r = img[:,w-strip:w,:]
    return [t, r, b, l]

def compute_features(strip):
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    lab_flat = lab.reshape(-1,3).astype(np.float32)

    hist = []
    for ch in range(3):
        hist.extend(np.histogram(lab_flat[:,ch], bins=16, range=(0,255))[0])
    hist = np.array(hist, dtype=np.float32)
    hist /= (np.linalg.norm(hist) + 1e-6)

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag_hist = np.histogram(mag, bins=16, range=(0,255))[0].astype(np.float32)
    mag_hist /= (np.linalg.norm(mag_hist) + 1e-6)

    return np.concatenate((hist, mag_hist))

def build_edge_features(pieces):
    ids = sorted(pieces.keys())
    imgs = {pid: cv2.imread(path) for pid, path in pieces.items()}

    example = next(iter(imgs.values()))
    h, w = example.shape[:2] # type: ignore
    strip = max(8, min(h,w) // 10)

    features = {}
    for pid in ids:
        strips = extract_edge_strip(imgs[pid], strip)
        features[pid] = []
        for s in strips:
            features[pid].append(compute_features(s))
    return ids, features
