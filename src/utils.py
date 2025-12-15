import cv2
import os

def save_image(img, img_name, output_dir, suffix=""):
    os.makedirs(output_dir, exist_ok=True)

    if suffix:
        filename = f"{img_name}_{suffix}.png"
    else:
        filename = f"{img_name}.png"

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img)
    return save_path