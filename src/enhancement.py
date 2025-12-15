import cv2
import numpy as np

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise_gaussian(gray):
    return cv2.GaussianBlur(gray, (3,3), sigmaX=0.5)

def denoise_bilateral(gray):
    return cv2.bilateralFilter(gray, d = 5,
sigmaColor = 25,
sigmaSpace = 10
)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10,10))
    return clahe.apply(gray)

def sharpen(img):
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=1.0)
    return cv2.addWeighted(img, 2, blur, -1, 0)

def morphology(img):
    kernel = np.ones((2,2), np.uint8)
    opened = img
    if np.mean(img) < 80:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def enhance_image(img):  
    gray = to_grayscale(img)
    denoised = denoise_bilateral(gray)
    
    clahe_img = apply_clahe(denoised)
    sharp_img_clahe = sharpen(clahe_img)

    sharp_img = sharpen(denoised)

    return sharp_img, sharp_img_clahe
