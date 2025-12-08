import numpy as np
import cv2

def reconstruct(placement, ids, pieces, output_path):
    h = cv2.imread(pieces[ids[0]]).shape[0] # type: ignore
    w = cv2.imread(pieces[ids[0]]).shape[1] # type: ignore
    grid = placement.shape[0]

    canvas = np.zeros((h*grid,w*grid,3), dtype=np.uint8)

    for r in range(grid):
        for c in range(grid):
            pid_idx = placement[r,c]
            pid = ids[pid_idx]
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = cv2.imread(pieces[pid])

    cv2.imwrite(output_path, canvas)
    print(f"✓ Saved: {output_path}")
