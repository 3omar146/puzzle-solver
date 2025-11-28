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
