import os
import cv2

from src.enhancement import enhance_image
from src.utils import save_image
from src.grid_detection import detect_grid_size
from src.segmentation import segment_and_extract

# Fixed output folders (since they are known)
ENHANCED_DIR    = "data/enhanced"
CONTOURS_DIR    = "data/contours"
PIECES_DIR      = "data/pieces"
DESCRIPTORS_DIR = "data/descriptors"

correct = 0
wrong = 0
total = 0

def process_single_image(img_path,grid_size,auto_detection):
    """
    Runs the enhancement step on a single image.
    Other steps (thresholding, segmentation, descriptors)
    are commented out until implemented.
    """
    grid_folder = f"{grid_size}x{grid_size}"

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    original = cv2.imread(img_path)

    print(f"\nProcessing {img_name}...")

    # -------------------------------
    # Step 1: Enhancement (ACTIVE)
    # -------------------------------
    enhanced, enhanced_clahe = enhance_image(original)
    save_image(enhanced, img_name,
               os.path.join(ENHANCED_DIR, grid_folder),
               suffix="enhanced")

    # -------------------------------
    # Step 2: Detect grid size (2x2, 4x4, 8x8)
    # -------------------------------
    if(auto_detection):
        grid_size = detect_grid_size(enhanced, enhanced_clahe)
        print(f"[INFO] Detected grid by the size detection algorithm: {grid_size}x{grid_size}")

    # -------------------------------
    # Step 3: Segmentation + Extraction (NOT READY)
    # -------------------------------
    contour_img, cropped_pieces, piece_metadata = segment_and_extract(
         original,grid_size, img_name
     )
    
    save_image(contour_img, img_name,
                os.path.join(CONTOURS_DIR, grid_folder),
                suffix="contours")    
    

    piece_folder = os.path.join(PIECES_DIR, grid_folder, img_name)
    os.makedirs(piece_folder, exist_ok=True)

    for piece_info, piece_img in zip(piece_metadata, cropped_pieces):
        save_image(piece_img, img_name,
                piece_folder,
                suffix=f"{piece_info['id']}") 


    # -------------------------------
    # Step 4: Descriptor generation (NOT READY)
    # -------------------------------
    # descriptor_dict = build_descriptor_dict(piece_metadata, cropped_pieces)
    # save_descriptor_json(descriptor_dict, img_name, DESCRIPTORS_DIR)

    print(f"Finished {img_name}")
    return grid_size



results = []

def process_dataset(dataset_folder, auto_detection):
    global correct, wrong, total

    correct = 0
    wrong = 0
    total = 0

    name = os.path.basename(dataset_folder)
    
    expected = None
    if "2x2" in name:
        expected = 2
    elif "4x4" in name:
        expected = 4
    elif "8x8" in name:
        expected = 8

    if expected is None:
        print(f"[ERROR] Folder name must contain grid size: {name}")
        return

    filenames = [f for f in os.listdir(dataset_folder)
                 if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

    print(f"Found {len(filenames)} images.")

    for filename in filenames:
        img_path = os.path.join(dataset_folder, filename)

        detected = process_single_image(img_path, expected, auto_detection)
        
        total += 1

        if detected is None:
            wrong += 1
        elif detected == expected:
            correct += 1
        else:
            wrong += 1

    results.append({
        "dataset": name,
        "correct": correct,
        "wrong": wrong,
        "total": total,
        "accuracy": (correct / total * 100) if total != 0 else 0
    })


def print_accuracy_table():
    print("\nFinal Accuracy Table:\n")
    print("{:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "Dataset", "Correct", "Wrong", "Total", "Accuracy"
    ))
    print("-" * 60)

    for r in results:
        print("{:<15} {:<10} {:<10} {:<10} {:<10.2f}".format(
            r["dataset"], r["correct"], r["wrong"], r["total"], r["accuracy"]
        ))
