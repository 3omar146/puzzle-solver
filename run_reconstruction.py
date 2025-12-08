import os
from src import paths
from src.compatibility import build_edge_features
from src.bestbuddies import compatibility_scores, best_buddies
from src.solver import solve_layout
from src.reconstruction import reconstruct
from src.image_accuracy import pixel_accuracy

def run(folder):
    g = int(folder.split("x")[0])
    root = os.path.join(paths.COLORED_PIECES_DIR, folder)
    out = "assembled_results"
    os.makedirs(out, exist_ok=True)

    results = []

    for img_id in sorted(os.listdir(root)):
        piece_dir = os.path.join(root, img_id)
        if not os.path.isdir(piece_dir): continue

        pieces = {
            int(f.split("_")[1].split(".")[0]): os.path.join(piece_dir,f)
            for f in os.listdir(piece_dir)
            if f.endswith(".png")
        }

        print(f"\nSolving {img_id} ({folder})...")

        ids, features = build_edge_features(pieces)
        comp    = compatibility_scores(ids, features)
        buddies = best_buddies(comp)
        placement, ids = solve_layout(ids, buddies, g)

        out_path = os.path.join(out, f"{img_id}_{folder}.png")
        reconstruct(placement, ids, pieces, out_path)

        truth = os.path.join("data/raw/Gravity Falls/correct", f"{img_id}.png")

        # Evaluate reconstruction via pixel accuracy
        acc, corr, total = pixel_accuracy(out_path, truth, g)
        print(f"Accuracy={acc:.2f}% ({corr}/{total} pixels correct)")
        results.append((img_id, acc, corr, total))

    # Save results summary file
    with open(os.path.join(out,f"{folder}_pixel_accuracy.txt"),"w") as f:
        for img_id,acc,corr,total in results:
            f.write(f"{img_id}: {acc:.2f}% ({corr}/{total} pixels correct)\n")


if __name__ == "__main__":
    run("2x2")
    run("4x4")
    run("8x8")
