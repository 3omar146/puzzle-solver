import cv2
import os

def save_image(img, img_name, output_dir, suffix=""):
    """
    Saves an image to the specified output directory with an optional suffix.
    Ensures the directory exists.
    
    Example filename: img001_binary.png or img001_contours.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if suffix:
        filename = f"{img_name}_{suffix}.png"
    else:
        filename = f"{img_name}.png"

    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, img)
    return save_path


def delete_images_in_folder(folder_path):
    """
    Deletes ALL image files (png, jpg, jpeg) inside the specified folder.
    Does NOT delete the folder itself.
    """
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                count += 1
            except Exception as e:
                print(f"Could not delete {file_path}: {e}")

    print(f"Deleted {count} image(s) from {folder_path}.")