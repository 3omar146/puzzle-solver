# step - 4:
# Takes the list of segmented puzzle pieces (from step 3) and generates descriptors for each piece.
#
# For each piece:
#   - Extract the full contour (precise shape boundary in the cropped piece)
#   - Split the contour into 4 edges (top, right, bottom, left)
#   - Store piece ID, bounding box, contour points, and edge point arrays
#
# Returns a structured descriptor dictionary containing ALL pieces for that image.
#
# The pipeline (not this file) is responsible for saving the JSON:
# Descriptor JSON is saved to: "data/descriptors/<image_name>.json"
#
# JSON Format:
# {
#   "pieces": [
#     {
#       "id": "img001_piece_00",
#       "bbox": [x, y, w, h],
#       "contour": [[...]],        # full contour points
#       "edges": {
#         "top":    [[...]],       # list of (x, y) points for top edge
#         "right":  [[...]],
#         "bottom": [[...]],
#         "left":   [[...]]
#       }
#     },
#     ...
#   ]
# }
