import os
import cv2

from src.enhancement import enhance_image
from src.utils import save_image

#rest of imports goes hena

# Fixed output folders (since they are known)
ENHANCED_DIR   = "data/enhanced"
BINARY_DIR     = "data/binary"
CONTOURS_DIR   = "data/contours"
PIECES_DIR     = "data/pieces"
DESCRIPTORS_DIR = "data/descriptors"


def process_single_image(img_path):
    """
    Runs the enhancement step on a single image.
    Other steps (thresholding, segmentation, descriptors)
    are commented out until implemented.
    """

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    original = cv2.imread(img_path)

    if original is None:
        print(f"ERROR: Could not load {img_path}")
        return

    print(f"\nProcessing {img_name} ...")

    # -------------------------------
    # Step 1: Enhancement (ACTIVE)
    # -------------------------------
    enhanced = enhance_image(original)
    save_image(enhanced, img_name, ENHANCED_DIR, suffix="enhanced")


    # -------------------------------
    # Step 2: Segmentation + Extraction (NOT READY)
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
    # Step 3: Descriptor generation (NOT READY)
    # -------------------------------
    # descriptor_dict = build_descriptor_dict(piece_metadata, cropped_pieces)
    # save_descriptor_json(descriptor_dict, img_name, DESCRIPTORS_DIR)

    print(f"Finished {img_name} ✓")


def process_dataset(dataset_folder):
    """
    Runs the pipeline (enhancement only for now) on every file
    in data/raw/ — all files are assumed to be valid images.
    """

    filenames = os.listdir(dataset_folder)
    print(f"Found {len(filenames)} images.")

    for filename in filenames:
        img_path = os.path.join(dataset_folder, filename)
        process_single_image(img_path)
