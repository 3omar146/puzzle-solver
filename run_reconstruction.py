import os
import cv2
from pathlib import Path

from src.reconstruction import (
    load_pieces,
    solve_2x2,
    solve_NxN,
    assemble_2x2,
    assemble_NxN,
)

from src.paths import EDGE_PIECES_DIR, RECONSTRUCTED_DIR

os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)

for grid in os.listdir(EDGE_PIECES_DIR):
    grid_path = Path(EDGE_PIECES_DIR) / grid
    if not grid_path.is_dir():
        continue

    N = int(grid.split("x")[0])

    for puzzle in grid_path.iterdir():
        if not puzzle.is_dir():
            continue

        print(f"Processing {grid}/{puzzle.name}")
        pieces = load_pieces(puzzle)

        if len(pieces) < N * N:
            print("Skipping (not enough pieces)")
            continue

        if N == 2:
            sol = solve_2x2(pieces)
            if sol is None:
                continue
            assembled = assemble_2x2(*sol)
        else:
            assembled = assemble_NxN(solve_NxN(pieces, N))

        out = Path(RECONSTRUCTED_DIR) / grid / f"{puzzle.name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), assembled)
        print("Saved:", out)
