import os
from src import paths
from src.compatibility import build_edge_features
from src.bestbuddies import compatibility_scores, best_buddies
from src.solver import solve_layout
from src.reconstruction import reconstruct

def run(folder):
    g = int(folder.split("x")[0])
    root = os.path.join(paths.COLORED_PIECES_DIR, folder)
    out = "assembled_results"
    os.makedirs(out, exist_ok=True)

    for img_id in os.listdir(root):
        piece_dir = os.path.join(root, img_id)
        if not os.path.isdir(piece_dir): continue

        pieces = {int(f.split("_")[1].split(".")[0]):
                    os.path.join(piece_dir,f)
                  for f in os.listdir(piece_dir) if f.endswith(".png")}

        print(f"\n[+] Solving {img_id} ({folder}) ...")

        ids, features = build_edge_features(pieces)
        comp = compatibility_scores(ids, features)
        buddies = best_buddies(comp)
        placement, ids = solve_layout(ids, buddies, g)

        out_path = os.path.join(out, f"{img_id}_{folder}.png")
        reconstruct(placement, ids, pieces, out_path)

if __name__ == "__main__":
    run("2x2")
    run("4x4")
    run("8x8")
