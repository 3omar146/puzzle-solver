import numpy as np

OPPOSITE = {0:2, 1:3, 2:0, 3:1}

def compatibility_scores(ids, features):
    N = len(ids)
    comp = np.ones((N,4,N)) * 1e9

    for i, pa in enumerate(ids):
        for e in range(4):
            fA = features[pa][e].reshape(1, -1)
            for j, pb in enumerate(ids):
                if pa == pb: continue
                fB = features[pb][OPPOSITE[e]].reshape(1, -1)
                dist = 1 - float(np.dot(fA,fB.T)/(np.linalg.norm(fA)*np.linalg.norm(fB)+1e-6))
                comp[i,e,j] = dist
    return comp

def best_buddies(comp):
    N = comp.shape[0]
    buddies = {i:{} for i in range(N)}

    for i in range(N):
        for e in range(4):
            j = np.argmin(comp[i,e])
            if j == i: continue
            e_op = np.argmin(comp[j,OPPOSITE[e]])
            if e_op == i:
                buddies[i][e] = j
    return buddies
