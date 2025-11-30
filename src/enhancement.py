import cv2
import numpy as np

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(gray):
    # VERY light smoothing ONLY to reduce tiny camera noise
    return cv2.GaussianBlur(gray, (3,3), sigmaX=0.5)

def apply_clahe(gray):
    # CLAHE causes patchiness on cartoon pieces — reduce it a lot
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def sharpen(img):
    # stronger edge sharpening but controlled
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)

def morphology(img):
    # light cleanup, avoids damaging cartoon shapes
    kernel = np.ones((2,2), np.uint8)
    opened = img
    if np.mean(img) < 80:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def enhance_image(img):
    gray      = to_grayscale(img)
    denoised  = denoise(gray)
    clahe_img = apply_clahe(denoised)

    sharp_img_clahe = sharpen(clahe_img)
    clean_img_clahe = morphology(sharp_img_clahe)
    
    sharp_img = sharpen(denoised)
    clean_img = morphology(sharp_img)

    return clean_img, clean_img_clahe
