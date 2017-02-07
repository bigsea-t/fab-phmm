import numpy as np

EPS = np.finfo(np.float).eps

def logsumexp(arr):
    vmax = arr.max()
    if vmax == -np.inf:
        return -np.inf
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


def tp(ans, pred):
    return np.logical_and(ans, pred).sum()


def tn(ans, pred):
    return np.logical_and(np.logical_not(ans), np.logical_not(pred)).sum()


def fp(ans, pred):
    return np.logical_and(np.logical_not(ans), pred).sum()


def fn(ans, pred):
    return np.logical_and(ans, np.logical_not(pred)).sum()


def gen_alignemnt_matrix(xseq_wgap, yseq_wgap):
    assert(xseq_wgap.shape[0] == yseq_wgap.shape[0])
    xlen = (xseq_wgap != -1).sum()
    ylen = (yseq_wgap != -1).sum()
    matrix = np.zeros((xlen + 1, ylen + 1), dtype=np.int_)

    i, j = 0, 0
    for x, y in zip(xseq_wgap, yseq_wgap):
        if x != -1:
            i += 1
        if y != -1:
            j += 1

        if x!= -1 and y != -1:
            matrix[i, j] = 1

    assert (i, j) == (xlen, ylen)

    return matrix

