import os
import cv2

from src.edge_detection import canny_edges
from src.enhancement import enhance_image
from src.utils import save_image
from src.size_detection import detect_grid_size
from src.segmentation import segment_and_extract
from src.thresholding import threshold_adaptive,threshold_otsu

# Fixed output folders (since they are known)
ENHANCED_DIR    = "data/enhanced"
CONTOURS_DIR    = "data/contours"
COLORED_PIECES_DIR   = "data/colored_pieces"
BINARY_PIECES_DIR   = "data/binary_pieces"
ENHANCED_PIECES_DIR = "data/enhanced_pieces"
EDGE_PIECES_DIR   = "data/edge_pieces"
DESCRIPTORS_DIR = "data/descriptors"
BINARY_DIR = "data/binary"
EDGES_DIR = "data/edges"


correct = 0
wrong = 0
total = 0

def process_single_image(img_path, grid_size, auto_detection):
    """
    1. Enhance entire image
    2. Detect grid size (if auto)
    3. Segment into pieces
    4. Threshold & edge detect per piece
    """

    grid_folder = f"{grid_size}x{grid_size}"
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    original = cv2.imread(img_path)

    print(f"\nProcessing {img_name}...")

    # Step 1: Enhance entire image
    enhanced, enhanced_clahe = enhance_image(original)
    save_image(enhanced, img_name,
               os.path.join(ENHANCED_DIR, grid_folder),
               suffix="enhanced")

    # Step 2: Detect grid size
    if auto_detection:
        grid_size = detect_grid_size(enhanced, enhanced_clahe)
        print(f"[INFO] Detected grid: {grid_size}x{grid_size}")

    # Step 3: Segment original & enhanced
    contour_img, cropped_pieces, piece_metadata = segment_and_extract(
        original, grid_size, img_name
    )

    save_image(contour_img, img_name,
               os.path.join(CONTOURS_DIR, grid_folder),
               suffix="contours")

    _, cropped_enhanced_pieces, enhanced_piece_metadata = segment_and_extract(
        enhanced_clahe, grid_size, img_name
    )

    colored_piece_folder = os.path.join(COLORED_PIECES_DIR, grid_folder, img_name)
    os.makedirs(colored_piece_folder, exist_ok=True)

    # Save original pieces
    for piece_info, piece_img in zip(piece_metadata, cropped_pieces):
        save_image(piece_img, img_name,
                   colored_piece_folder,
                   suffix=f"{piece_info['id']}")
    
    # Save enhanced pieces
    colored_piece_folder = os.path.join(ENHANCED_PIECES_DIR, grid_folder, img_name)
    os.makedirs(colored_piece_folder, exist_ok=True)

    for piece_info, piece_img in zip(enhanced_piece_metadata, cropped_enhanced_pieces):
        save_image(piece_img, img_name,
                   colored_piece_folder,
                   suffix=f"{piece_info['id']}")

    # Step 4: Threshold & edge-detect per piece
    binary_piece_folder = os.path.join(BINARY_PIECES_DIR, grid_folder, img_name)
    edge_piece_folder   = os.path.join(EDGE_PIECES_DIR, grid_folder, img_name)

    os.makedirs(binary_piece_folder, exist_ok=True)
    os.makedirs(edge_piece_folder, exist_ok=True)

    for piece_info, piece_img in zip(enhanced_piece_metadata, cropped_enhanced_pieces):
        # Threshold
        binary_piece = threshold_adaptive(piece_img)
        save_image(binary_piece, img_name,
                   binary_piece_folder,
                   suffix=f"{piece_info['id']}")

        # Edge-detect
        edge_piece = canny_edges(piece_img)
        save_image(edge_piece, img_name,
                   edge_piece_folder,
                   suffix=f"{piece_info['id']}")

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

    filenames = os.listdir(dataset_folder)

    for filename in filenames:
        img_path = os.path.join(dataset_folder, filename)

        detected = process_single_image(img_path, expected, auto_detection)
        
        if auto_detection:
            total += 1

            if detected is None: wrong += 1
            elif detected == expected: correct += 1
            else: wrong += 1

    if auto_detection:
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
