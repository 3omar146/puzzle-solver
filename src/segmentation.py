# step - 3:
# Takes the ORIGINAL colored image and the BINARY image.
# Detects external contours to locate puzzle pieces and creates a contour-visualized image.
#
# Filters the detected contours using:
#   - area thresholds (to remove noise and borders)
#   - aspect ratio (to eliminate long thin grid lines)
#   - solidity (to remove hollow or irregular shapes)
# This keeps ONLY contours corresponding to actual puzzle pieces.
#
# For each valid puzzle-piece contour:
#   - Crop the piece from the ORIGINAL colored image using its bounding box
#
# Returns:
#   1. The contour-visualized image (to be saved in pipeline to: "data/contours/")
#   2. A list of cropped RGB piece images (np arrays) for saving in pipeline under:
#        "data/pieces/<image_name>/"
#   3. A list of metadata for each piece:
#        { "id", "bbox", "contour" }
#      which is passed to Step 4 (descriptor extraction)
import os
import cv2



def segment_and_extract(original_img, grid_size, img_name):
    """
    HARD CODED SEGMENTATION:
    Takes the image and splits it into grid_size x grid_size tiles.
    Saves tiles inside: data/pieces/<img_name>/
    Returns (contour_img, cropped_pieces, piece_metadata)
    """

    h, w = original_img.shape[:2]

    # Calculate size of each tile
    tile_h = h // grid_size
    tile_w = w // grid_size


    contour_img = original_img.copy()
    cropped_pieces = []
    piece_metadata = []

    piece_id = 1

    for r in range(grid_size):
        for c in range(grid_size):
            # Coordinates of the tile
            y1 = r * tile_h
            y2 = (r + 1) * tile_h
            x1 = c * tile_w
            x2 = (c + 1) * tile_w

            # Crop tile
            tile = original_img[y1:y2, x1:x2]


            # Add to output lists
            cropped_pieces.append(tile)
            piece_metadata.append({
                "id": piece_id,
                "row": r,
                "col": c,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

            # Draw bounding box on contour image
            cv2.rectangle(contour_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            piece_id += 1

    return contour_img, cropped_pieces, piece_metadata