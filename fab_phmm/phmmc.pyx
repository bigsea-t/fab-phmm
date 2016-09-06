# cython: boundscheck=False, wraparound=False

import numpy as np
from numpy.math cimport isinf, INFINITY, expl, logl

ctypedef double dtype_t

cdef inline int _argmax(dtype_t[:] arr) nogil:
    cdef dtype_t max_x = - INFINITY
    cdef int pos = 0, i

    for i in range(arr.shape[0]):
        if arr[i] > max_x:
            pos = i
            max_x = arr[i]

    return pos

cdef inline dtype_t _max(dtype_t[:] arr) nogil:
    return arr[_argmax(arr)]

cdef inline dtype_t _logsumexp(dtype_t[:] arr) nogil:
    cdef dtype_t max_x = _max(arr)

    if isinf(max_x):
        return - INFINITY

    cdef dtype_t acc = 0
    cdef int i

    for i in range(arr.shape[0]):
        acc += expl(arr[i] - max_x)

    return logl(acc) + max_x

cdef int[:] dx = np.array([1, 1, 0], dtype=np.int32)
cdef int[:] dy = np.array([1, 0 ,1], dtype=np.int32)

def _forward(int shape_x,
             int shape_y,
             int n_hstates,
             int[:] hstate_properties,
             dtype_t[:, :, :] log_emitprob_frame,
             dtype_t[:] log_initprob,
             dtype_t[:, :] log_transprob,
             dtype_t[:, :, :] fwd_lattice):

    cdef int t, u, r, s, j, k, p

    cdef dtype_t[:] wbuf = np.zeros(n_hstates)

    #with nogil:

    for k in range(n_hstates):
        p = hstate_properties[k]
        fwd_lattice[dx[p], dy[p], k] = log_emitprob_frame[dx[p], dy[p], k] + log_initprob[k]

    for t in range(shape_x):
        for u in range(shape_y):

            for k in range(n_hstates):
                p = hstate_properties[k]
                r = t - dx[p]
                s = u - dy[p]

                if (r == 0 and s == 0) or r < 0 or s < 0:
                    continue

                for j in range(n_hstates):
                    wbuf[j] = fwd_lattice[r, s, j] + log_transprob[j, k]

                fwd_lattice[t, u, k] = log_emitprob_frame[t, u, k] + _logsumexp(wbuf)

def _backward(int shape_x,
              int shape_y,
              int n_hstates,
              int[:] hstate_properties,
              dtype_t[:, :, :] log_emitprob_frame,
              dtype_t[:, :] log_transprob,
              dtype_t[:, :, :] bwd_lattice):

    cdef int t, u, r, s, k, l, p

    cdef dtype_t[:] wbuf = np.zeros(n_hstates)

    with nogil:

        for k in reversed(range(n_hstates)):
            bwd_lattice[shape_x - 1, shape_y - 1, k] = 0

        for t in reversed(range(shape_x)):
            for u in range(shape_y):

                if t == shape_x - 1 and u == shape_y - 1:
                     continue

                for k in range(n_hstates):
                    for l in range(n_hstates):
                        p = hstate_properties[k]
                        r = t - dx[p]
                        s = u - dy[p]

                        if r >= shape_x or s >= shape_y:
                            wbuf[l] = - INFINITY
                        else:
                            wbuf[l] = bwd_lattice[r, s, l] + log_transprob[k, l] \
                                + log_emitprob_frame[r, s, l]

                        bwd_lattice[t, u, k] = _logsumexp(wbuf)










