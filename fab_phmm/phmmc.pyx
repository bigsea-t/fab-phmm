# cython: boundscheck=False, wraparound=False

import numpy as np
from numpy.math cimport isinf, INFINITY, expl, logl

ctypedef double dtype_t
ctypedef long int_t

cdef inline int_t _argmax(dtype_t[:] arr) nogil:
    cdef dtype_t max_x = - INFINITY
    cdef int_t pos = 0, i

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
    cdef int_t i

    for i in range(arr.shape[0]):
        acc += expl(arr[i] - max_x)

    return logl(acc) + max_x


def _forward(int_t shape_x,
             int_t shape_y,
             int_t n_hstates,
             int_t[:] hstate_properties,
             dtype_t[:, :, :] log_emitprob_frame,
             dtype_t[:] log_initprob,
             dtype_t[:, :] log_transprob,
             dtype_t[:, :, :] fwd_lattice):

    cdef int_t t, u, r, s, j, k, p

    cdef int_t[:] dx = np.array([1, 1 ,0], dtype=np.int_)
    cdef int_t[:] dy = np.array([1, 0 ,1], dtype=np.int_)

    cdef dtype_t[:] wbuf = np.zeros(n_hstates)

    with nogil:

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


def _backward(int_t shape_x,
              int_t shape_y,
              int_t n_hstates,
              int_t[:] hstate_properties,
              dtype_t[:, :, :] log_emitprob_frame,
              dtype_t[:, :] log_transprob,
              dtype_t[:, :, :] bwd_lattice):

    cdef int_t t, u, r, s, k, l, p

    cdef int_t[:] dx = np.array([1, 1 ,0], dtype=np.int_)
    cdef int_t[:] dy = np.array([1, 0 ,1], dtype=np.int_)

    # 2-d wbuf to avoid cache miss
    cdef dtype_t [:, :] wbuf = np.zeros((n_hstates, n_hstates), dtype=np.float64)

    with nogil:

        for k in range(n_hstates):
            bwd_lattice[shape_x - 1, shape_y - 1, k] = 0

        for t in reversed(range(shape_x)):
            for u in reversed(range(shape_y)):

                if t == shape_x - 1 and u == shape_y - 1:
                    continue

                if t == 0 and u == 0:
                    continue

                for l in range(n_hstates):
                    for k in range(n_hstates):
                        p = hstate_properties[l]
                        r = t + dx[p]
                        s = u + dy[p]

                        if r >= shape_x or s >= shape_y:
                            wbuf[k, l] = - INFINITY
                        else:
                            wbuf[k, l] = bwd_lattice[r, s, l] + log_transprob[k, l] \
                                + log_emitprob_frame[r, s, l]

                for k in range(n_hstates):
                    bwd_lattice[t, u, k] = _logsumexp(wbuf[k, :])


def _accumulate_emitprob_match(dtype_t[:, :, :] sstats_emit,
                         dtype_t[:, :, :] gamma,
                         int_t[:] xseq, int_t[:] yseq,
                         int_t len_xseq, int_t len_yseq, int_t k):
    cdef int_t t, u;
    for t in range(len_xseq):
        for u in range(len_yseq):
            sstats_emit[k, xseq[t], yseq[u]] = gamma[t+1, u+1, k] + sstats_emit[k, xseq[t], yseq[u]]


def _accumulate_emitprob_xins(dtype_t[:, :, :] sstats_emit,
                         dtype_t[:, :, :] gamma,
                         int_t[:] xseq, int_t[:] yseq,
                         int_t len_xseq, int_t len_yseq, int_t k):
    cdef int_t t, u;
    for t in range(len_xseq):
        for u in range(len_yseq+1):
            sstats_emit[k, xseq[t], :] = gamma[t+1, u, k] + sstats_emit[k, xseq[t], 0]


def _accumulate_emitprob_yins(dtype_t[:, :, :] sstats_emit,
                         dtype_t[:, :, :] gamma,
                         int_t[:] xseq, int_t[:] yseq,
                         int_t len_xseq, int_t len_yseq, int_t k):
    cdef int_t t, u;
    for t in range(len_xseq+1):
        for u in range(len_yseq):
            sstats_emit[k, :, yseq[u]] = gamma[t, u+1, k] + sstats_emit[k, 0, yseq[u]]


def _compute_log_emitprob_frame(dtype_t[:, :, :] log_emitprob_frame,
                                dtype_t[:, :, :] log_emitprob,
                                int_t[:] xseq, int_t[:] yseq,
                                int_t[:] hstate_props,
                                int_t len_x, int_t len_y, int_t n_hstates):
    cdef int_t i, j, k;

    for i in range(len_x):
        for j in range(len_y):
            for k in range(n_hstates):
                log_emitprob_frame[i + 1, j + 1, k] = log_emitprob[k, xseq[i], yseq[j]]

    for k in range(n_hstates):
        if hstate_props[k] == 1:
            for i in range(len_x):
                log_emitprob_frame[i + 1, 0, k] = log_emitprob[k, xseq[i], 0]

    for k in range(n_hstates):
        if hstate_props[k] == 2:
            for j in range(len_y):
                log_emitprob_frame[0, j + 1, k] = log_emitprob[k, 0, yseq[j]]


def _compute_two_sliced_margnial(int_t shape_x,
                                 int_t shape_y,
                                 int_t n_hstates,
                                 int_t[:] hstate_properties,
                                 dtype_t ll,
                                 dtype_t[:, :, :] fwd_lattice,
                                 dtype_t[:, :, :] bwd_lattice,
                                 dtype_t[:, :, :] log_emitprob_frame,
                                 dtype_t[:, :] log_transprob,
                                 dtype_t[:, :, :, :] log_xi):

    cdef int_t[:] dx = np.array([1, 1 ,0], dtype=np.int_)
    cdef int_t[:] dy = np.array([1, 0 ,1], dtype=np.int_)

    cdef int_t t, u, j, k, dt, du, p;

    with nogil:
        for t in range(shape_x):
            for u in range(shape_y):
                for k in range(n_hstates):
                    p = hstate_properties[k]
                    dt = dx[p]
                    du = dy[p]

                    if t + dt >= shape_x or u + du >= shape_y:
                        continue

                    for j in range(n_hstates):
                        log_xi[t, u, j, k] = fwd_lattice[t, u, j] + \
                                 log_emitprob_frame[t + dt, u + du, k] + \
                                 log_transprob[j, k] + \
                                 bwd_lattice[t + dt, u + du, k] - ll

def _decode(int_t shape_x,
    int_t shape_y,
    int_t n_hstates,
    int_t[:] hstate_properties,
    dtype_t[:, :, :] log_emitprob_frame,
    dtype_t[:] log_initprob,
    dtype_t[:, :] log_transprob,
    dtype_t[:, :, :] viterbi_lattice,
    int_t [:] map_hstates):

    cdef int_t t, u, r, s, k, l, opt_l, i, c, tmp, len_seq, p, xins_i, yins_i
    cdef dtype_t ll

    cdef int_t[:] dx = np.array([1, 1 ,0], dtype=np.int_)
    cdef int_t[:] dy = np.array([1, 0 ,1], dtype=np.int_)

    cdef dtype_t [:] wbuf = np.zeros(n_hstates, dtype=np.float64)

    cdef int_t[:, :, :] trace_lattice = np.zeros((shape_x, shape_y, n_hstates), dtype=np.int_)

    xins_i = list(hstate_properties).index(1)
    yins_i = list(hstate_properties).index(2)

    with nogil:

        for k in range(n_hstates):
            p = hstate_properties[k]
            viterbi_lattice[dx[p], dy[p], k] = log_emitprob_frame[0, 0, k] + log_initprob[k]
            trace_lattice[dx[p], dy[p], k] = k

        for t in range(shape_x):
            for u in range(shape_y):

                if t == 0 and u == 0:
                    continue

                for k in range(n_hstates):
                    if not isinf(viterbi_lattice[t, u, k]):
                        continue

                    p = hstate_properties[k]

                    r = t - dx[p]
                    s = u - dy[p]

                    for l in range(n_hstates):
                        wbuf[l] = - INFINITY

                    if r >= 0 and s >= 0:
                        for l in range(n_hstates):
                            wbuf[l] = viterbi_lattice[r, s, l] + log_transprob[l, k]

                    opt_l = _argmax(wbuf)
                    viterbi_lattice[t, u, k] = wbuf[opt_l] + log_emitprob_frame[t, u, k]
                    trace_lattice[t, u, k] = opt_l

        t = shape_x - 1
        u = shape_y - 1

        i = 0
        map_hstates[i] = _argmax(viterbi_lattice[t, u, :])
        ll = viterbi_lattice[t, u, map_hstates[i]]

        # for now return dummy

        if isinf(ll):
            for j in range(0, t):
                map_hstates[j] = xins_i
            for j in range(t, t + u):
                map_hstates[j] = yins_i
        else:
            while not (t == 0 and u == 0):
                i += 1
                c = map_hstates[i-1]
                map_hstates[i] = trace_lattice[t, u, c]
                p = hstate_properties[c]
                t -= dx[p]
                u -= dy[p]

            # reverse (consider -1 padding)

            # disregard the lastly appended element
            map_hstates[i] = -1
            len_seq = i

            #TODO: reverse has a bug
            for i in range(len_seq // 2):
                tmp = map_hstates[i]
                map_hstates[i] = map_hstates[len_seq - i - 1]
                map_hstates[len_seq - i - 1] = tmp

    return ll