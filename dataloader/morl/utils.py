import numpy as np
from math import isclose

isCloseToOne = lambda x: isclose(x, 1, rel_tol=1e-12)

def pref_grid(n_obj, max_prefs=None, min_prefs=None, granularity=5):
    max_prefs = np.ones(n_obj) if max_prefs is None else max_prefs
    min_prefs = np.zeros(n_obj) if min_prefs is None else min_prefs
    grid = np.array([x/granularity for x in range(granularity+1)])
    prefs = [[]]
    grid = tuple(grid)
    for _ in range(n_obj):
        prefs = [x+[y] for x in prefs for y in grid if sum(x+[y]) < 1 or isCloseToOne(sum(x+[y]))]
    prefs = np.array([p for p in prefs if isCloseToOne(sum(p))])
    for i in range(n_obj):
        prefs[:, i] = prefs[:, i] * (max_prefs[i] - min_prefs[i]) + min_prefs[i]
    prefs = prefs / np.sum(prefs, axis=1, keepdims=True)
    return prefs