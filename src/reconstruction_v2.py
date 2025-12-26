import cv2
import numpy as np

HORIZONTAL = "horizontal"
VERTICAL = "vertical"

# Seam cost computation
def compute_seam_costs(lab_pieces):
    left_edges = np.stack([p[:, 0] for p in lab_pieces])
    right_edges = np.stack([p[:, -1] for p in lab_pieces])
    top_edges = np.stack([p[0] for p in lab_pieces])
    bottom_edges = np.stack([p[-1] for p in lab_pieces])

    left_l_std = left_edges[..., 0].std(axis=1)
    right_l_std = right_edges[..., 0].std(axis=1)
    top_l_std = top_edges[..., 0].std(axis=1)
    bottom_l_std = bottom_edges[..., 0].std(axis=1)

    horizontal_cost = np.mean(
        np.abs(right_edges[:, None] - left_edges[None]), axis=(2, 3)
    )
    vertical_cost = np.mean(
        np.abs(bottom_edges[:, None] - top_edges[None]), axis=(2, 3)
    )

    horizontal_variance = 0.5 * (right_l_std[:, None] + left_l_std[None])
    vertical_variance = 0.5 * (bottom_l_std[:, None] + top_l_std[None])

    np.fill_diagonal(horizontal_cost, np.inf)
    np.fill_diagonal(vertical_cost, np.inf)

    return (
        (horizontal_cost, horizontal_variance),
        (vertical_cost, vertical_variance),
    )


# Edge scoring
def score_edges(horizontal_data, vertical_data):
    horizontal_cost, horizontal_variance = horizontal_data
    vertical_cost, vertical_variance = vertical_data

    num_pieces = horizontal_cost.shape[0]
    scored_edges = []

    # Adaptive variance scale
    all_variances = np.concatenate(
        [
            horizontal_variance[np.isfinite(horizontal_variance)],
            vertical_variance[np.isfinite(vertical_variance)],
        ]
    )
    variance_scale = np.median(all_variances) if len(all_variances) > 0 else 8.0

    VAR_WEIGHT = 0.5 # strength of variance penalty

    # Incoming-best (mutual match detection)
    best_horizontal_incoming = np.argmin(horizontal_cost, axis=0)
    best_vertical_incoming = np.argmin(vertical_cost, axis=0)

    for piece_idx in range(num_pieces):
        for cost_matrix, variance_matrix, orientation, best_incoming in (
            (horizontal_cost, horizontal_variance, HORIZONTAL, best_horizontal_incoming),
            (vertical_cost, vertical_variance, VERTICAL, best_vertical_incoming),
        ):
            cost_row = cost_matrix[piece_idx]

            # Find best and second-best matches
            first, second = np.argpartition(cost_row, 2)[:2]
            best_match, runner_up = (
                (first, second)
                if cost_row[first] <= cost_row[second]
                else (second, first)
            )

            best_cost = cost_row[best_match]
            second_cost = cost_row[runner_up]

            score = best_cost / second_cost

            # Mutual-best bonus
            if best_incoming[best_match] == piece_idx:
                score -= 0.5

            # Smooth variance-based penalty
            var = variance_matrix[piece_idx, best_match]
            score += VAR_WEIGHT * np.exp(-var / variance_scale)

            scored_edges.append((score, piece_idx, best_match, orientation))

    scored_edges.sort(key=lambda x: x[0])
    return scored_edges


class Cluster:
    __slots__ = ("grid_positions", "piece_positions")

    def __init__(self, piece_index):
        self.grid_positions = {(0, 0): piece_index}
        self.piece_positions = {piece_index: (0, 0)}

    def try_merge(self, other, src_idx, dst_idx, orientation, grid_size):
        src_row, src_col = self.piece_positions[src_idx]
        dst_row, dst_col = other.piece_positions[dst_idx]

        if orientation == HORIZONTAL:
            row_offset = src_row - dst_row
            col_offset = src_col + 1 - dst_col
        else:
            row_offset = src_row + 1 - dst_row
            col_offset = src_col - dst_col

        merged_positions = dict(self.grid_positions)
        merged_piece_positions = dict(self.piece_positions)

        for (row, col), piece_idx in other.grid_positions.items():
            new_row = row + row_offset
            new_col = col + col_offset

            if (new_row, new_col) in merged_positions:
                return False

            merged_positions[(new_row, new_col)] = piece_idx
            merged_piece_positions[piece_idx] = (new_row, new_col)

        rows = [r for r, _ in merged_positions]
        cols = [c for _, c in merged_positions]

        if max(rows) - min(rows) >= grid_size:
            return False
        if max(cols) - min(cols) >= grid_size:
            return False

        self.grid_positions = merged_positions
        self.piece_positions = merged_piece_positions
        return True


def start_reconstruction_v2(color_pieces, grid_size):
    num_pieces = len(color_pieces)

    lab_pieces = [
        cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32) for p in color_pieces
    ]

    horizontal_data, vertical_data = compute_seam_costs(lab_pieces)
    scored_edges = score_edges(horizontal_data, vertical_data)

    clusters = [Cluster(i) for i in range(num_pieces)]
    piece_owner = {i: clusters[i] for i in range(num_pieces)}

    for _, src_idx, dst_idx, orientation in scored_edges:
        cluster_a = piece_owner[src_idx]
        cluster_b = piece_owner[dst_idx]

        if cluster_a is cluster_b:
            continue

        if cluster_a.try_merge(cluster_b, src_idx, dst_idx, orientation, grid_size):
            for idx in list(cluster_b.piece_positions.keys()):
                piece_owner[idx] = cluster_a

            cluster_b.grid_positions.clear()
            cluster_b.piece_positions.clear()

    root_clusters = {
        piece_owner[i] for i in piece_owner if piece_owner[i].grid_positions
    }
    largest_cluster = max(root_clusters, key=lambda c: len(c.grid_positions))

    rows = [r for (r, _) in largest_cluster.grid_positions]
    cols = [c for (_, c) in largest_cluster.grid_positions]
    min_row, min_col = min(rows), min(cols)

    normalized_positions = {}
    for (row, col), idx in largest_cluster.grid_positions.items():
        new_row = row - min_row
        new_col = col - min_col

        normalized_positions[(new_row, new_col)] = idx

    largest_cluster.grid_positions = normalized_positions

    used_indices = set(largest_cluster.grid_positions.values())
    remaining_indices = [i for i in range(num_pieces) if i not in used_indices]

    if remaining_indices:
        occupied_slots = set(largest_cluster.grid_positions)
        free_slots = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if (r, c) not in occupied_slots
        ]

        for idx, slot in zip(remaining_indices, free_slots):
            largest_cluster.grid_positions[slot] = idx

    piece_height, piece_width = color_pieces[0].shape[:2]
    canvas = np.zeros((grid_size * piece_height, grid_size * piece_width, 3), np.uint8)

    for (row, col), idx in largest_cluster.grid_positions.items():
        canvas[
            row * piece_height : (row + 1) * piece_height,
            col * piece_width : (col + 1) * piece_width,
        ] = color_pieces[idx]

    return canvas
