import numpy as np

EPS = np.finfo(np.float).eps
INF = np.finfo(np.float).max
MINF = np.finfo(np.float).min


def logsumexp(arr):
    vmax = arr.max()
    out = np.log(np.sum(np.exp(arr - vmax)))
    out += vmax
    return out


def log_(arr):
    out = np.zeros_like(arr)
    zero_indices = arr == 0
    nonzero_indices = (zero_indices == False)
    out[zero_indices] = MINF / 10
    out[nonzero_indices] = np.log(arr[nonzero_indices])
    return out