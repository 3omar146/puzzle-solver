import numpy as np
from collections import deque
from src.bestbuddies import OPPOSITE

def solve_layout(ids, buddies, grid):
    N = grid*grid
    placement = np.full((grid,grid), -1, int)

    # pick seed: highest buddy count
    seed = max(buddies.keys(), key=lambda k: len(buddies[k]))
    placement[0,0] = seed
    placed = {seed}
    queue = deque([(0,0,seed)])

    while queue:
        r,c,p = queue.popleft()
        for e, nb in buddies[p].items():
            if nb in placed: continue
            dr,dc = [( -1,0),(0,1),(1,0),(0,-1 )][e]
            nr, nc = r+dr, c+dc
            if 0 <= nr < grid and 0 <= nc < grid and placement[nr,nc] == -1:
                placement[nr,nc] = nb
                placed.add(nb)
                queue.append((nr,nc,nb))

    # Fill any remaining cells from unused IDs
    unused = [i for i in range(N) if i not in placed]
    k=0
    for r in range(grid):
        for c in range(grid):
            if placement[r,c] == -1:
                placement[r,c] = unused[k]
                k+=1

    return placement, ids
