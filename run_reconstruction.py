import os
import cv2
from pathlib import Path

from src.paths import COLORED_PIECES_DIR, RECONSTRUCTED_DIR

from src.reconstruction_v1 import start_reconstruction_v1
from src.reconstruction_v2 import start_reconstruction_v2

RECONSTRUCTED_V2 = True

os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)

for grid in os.listdir(COLORED_PIECES_DIR):
    grid_path = Path(COLORED_PIECES_DIR) / grid
    if not grid_path.is_dir():
        continue

    N = int(grid.split("x")[0])

    for puzzle_dir in grid_path.iterdir():
        if not puzzle_dir.is_dir():
            continue

        print(f"Reconstructing {grid}/{puzzle_dir.name}")

        # Load pieces
        color_pieces = []
        for img_path in sorted(puzzle_dir.glob("*.png")):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is not None:
                color_pieces.append(img)

        if len(color_pieces) != N * N:
            print(f"Skipping (expected {N*N}, got {len(color_pieces)})")
            continue
        
        # Reconstruct
        if RECONSTRUCTED_V2:
            assembled = start_reconstruction_v2(color_pieces, N)
        else:
            assembled = start_reconstruction_v1(color_pieces, N)

        # Save result
        out = Path(RECONSTRUCTED_DIR) / grid
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{puzzle_dir.name}.png"

        cv2.imwrite(str(out_path), assembled) # type: ignore
        print("Saved:", out_path)
