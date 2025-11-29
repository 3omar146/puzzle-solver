import os
import cv2

from src.enhancement import enhance_image
from src.utils import save_image
from src.grid_detection import detect_grid_size

# Fixed output folders (since they are known)
ENHANCED_DIR    = "data/enhanced"
CONTOURS_DIR    = "data/contours"
PIECES_DIR      = "data/pieces"
DESCRIPTORS_DIR = "data/descriptors"

correct = 0
wrong = 0
total = 0

def process_single_image(img_path):
    """
    Runs the enhancement step on a single image.
    Other steps (thresholding, segmentation, descriptors)
    are commented out until implemented.
    """
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    original = cv2.imread(img_path)

    print(f"\nProcessing {img_name}...")

    # -------------------------------
    # Step 1: Enhancement (ACTIVE)
    # -------------------------------
    enhanced = enhance_image(original)
    save_image(enhanced, img_name, ENHANCED_DIR, suffix="enhanced")

    # -------------------------------
    # Step 2: Detect grid size (2x2, 4x4, 8x8)
    # -------------------------------
    grid_size = detect_grid_size(enhanced)
    print(f"[INFO] Detected grid: {grid_size}x{grid_size}")

    # -------------------------------
    # Step 3: Segmentation + Extraction (NOT READY)
    # -------------------------------
    # contour_img, cropped_pieces, piece_metadata = segment_and_extract(
    #     original, binary, img_name
    # )
    #
    # save_image(contour_img, img_name, CONTOURS_DIR, suffix="contours")
    #
    # piece_folder = os.path.join(PIECES_DIR, img_name)
    # os.makedirs(piece_folder, exist_ok=True)
    #
    # for piece_info, piece_img in zip(piece_metadata, cropped_pieces):
    #     cv2.imwrite(os.path.join(piece_folder, f"{piece_info['id']}.png"), piece_img)


    # -------------------------------
    # Step 4: Descriptor generation (NOT READY)
    # -------------------------------
    # descriptor_dict = build_descriptor_dict(piece_metadata, cropped_pieces)
    # save_descriptor_json(descriptor_dict, img_name, DESCRIPTORS_DIR)


    print(f"Finished {img_name} ✓")
    return grid_size

results = []

def process_dataset(dataset_folder):
    global correct, wrong, total

    correct = 0
    wrong = 0
    total = 0

    name = os.path.basename(dataset_folder)
    if "2x2" in name:
        expected = 2
    if "4x4" in name:
        expected = 4
    if "8x8" in name:
        expected = 8

    filenames = os.listdir(dataset_folder)
    print(f"Found {len(filenames)} images.")

    for filename in filenames:
        img_path = os.path.join(dataset_folder, filename)

        detected = process_single_image(img_path)
        if detected is None:
            continue

        total += 1
        if detected == expected:
            correct += 1
        else:
            wrong += 1

    results.append({
        "dataset": name,
        "correct": correct,
        "wrong": wrong,
        "total": total,
        "accuracy": (correct/total)*100
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
