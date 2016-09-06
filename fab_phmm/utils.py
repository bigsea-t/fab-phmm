import numpy as np

EPS = np.finfo(np.float).eps

def logsumexp(arr):
    vmax = arr.max()
    out = np.log(np.sum(np.exp(arr - vmax)))
    out += vmax
    return out


def log_(arr):
    out = np.zeros_like(arr)
    zero_indices = arr == 0
    nonzero_indices = (zero_indices == False)
    out[zero_indices] = -np.inf
    out[nonzero_indices] = np.log(arr[nonzero_indices])
    return out

def omit_gap(seq, gap_id=-1):
    out = []

    for a in seq:
        if a != gap_id:
            out.append(a)

    return np.array(out)