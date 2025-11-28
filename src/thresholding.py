# step - 2:
# Takes the enhanced grayscale image and applies adaptive thresholding to produce
# a binary (black & white) mask where puzzle pieces appear as white regions.
#
# This binary mask is essential for Step 3 (segmentation), since contour detection
# operates on binary shapes only.
#
# Returns the binary image to be saved by the pipeline under: "data/binary/"
