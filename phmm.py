import numpy as np
from utils import *


class PHMM:

    def __init__(self, n_match_states = 1, n_ins_states = 2, n_simbols=4, initprob=None, transprob=None, emitprob=None):
        self._initprob = initprob # [n_hstates]
        self._transprob = transprob # [n_hstates, n_hstates]
        self._emitprob = emitprob # [n_hstates, xdim, ydim] (usually xdim == ydim)

        self._n_hstates = n_match_states + n_ins_states * 2
        self._hstate_properties = self._gen_hstate_properties(n_match_states, n_ins_states)
        self._delta_index_list = [(1, 1), (1, 0), (0, 1)]

        self._n_simbols = n_simbols

        if initprob is None or transprob is None or emitprob is None:
            self._params_valid = False

    def _gen_hstate_properties(self, n_match_states, n_ins_states):
        # 0: Match, 1: Xins, 2: Yins
        hstate_properties = []
        for _ in range(n_match_states):
            hstate_properties.append(0)
        for _ in range(n_ins_states):
            hstate_properties.append(1)
        for _ in range(n_ins_states):
            hstate_properties.append(2)

        return hstate_properties

    def _delta_index(self, hstate):
        if hstate < 0 or hstate >= self._n_hstates:
            raise ValueError
        return self._delta_index_list[self._hstate_properties[hstate]]

    def _gen_log_emitprob_frame(self, xseq, yseq):
        len_x = xseq.shape[0]
        len_y = yseq.shape[0]
        shape = (len_x + 1, len_y + 1, self._n_hstates)
        log_emitprob = np.log(self._emitprob)
        log_emitprob_frame = np.zeros(shape)

        for i in range(len_x):
            for j in range(len_y):
                for k in range(self._n_hstates):
                    log_emitprob_frame[i + 1, j + 1, k] = log_emitprob[k, xseq[i], yseq[j]]

        for i in range(len_x):
            for k in range(self._n_hstates):
                if self._hstate_properties[k] == 1:
                    log_emitprob_frame[i + 1, 0, k] = log_emitprob[k, xseq[i], 0]

        for j in range(len_y):
            for k in range(self._n_hstates):
                if self._hstate_properties[k] == 2:
                    log_emitprob_frame[0, j + 1, k] = log_emitprob[k, 0, yseq[j]]

        return log_emitprob_frame

    def decode(self, x_seq, y_seq):
        log_initprob = np.log(self._initprob + EPS)
        log_transprob = np.log(self._transprob + EPS)

        len_x = x_seq.shape[0]
        len_y = y_seq.shape[0]

        lattice_shape = (len_x + 1, len_y + 1, self._n_hstates)
        viterbi_lattice = np.ones(lattice_shape) * MINF
        trace_lattice = np.ones(lattice_shape, dtype=np.int32) * (- 1)

        log_emitprob_frame = self._gen_log_emitprob_frame(x_seq, y_seq)

        for k in range(self._n_hstates):
            di, dj = self._delta_index(k)
            viterbi_lattice[di, dj, k] = log_emitprob_frame[0, 0, k] + log_initprob[k]
            trace_lattice[di, dj, k] = k

        # induction
        for i in range(len_x+1):
            for j in range(len_y+1):

                if i == 0 and j == 0:
                    continue

                for k in range(self._n_hstates):
                    if viterbi_lattice[i, j, k] != MINF:
                        continue

                    cands = np.ones(self._n_hstates) * MINF
                    di, dj = self._delta_index(k)
                    _i , _j = i - di, j - dj
                    if _i >= 0 and _j >= 0:
                        for l in range(self._n_hstates):
                            cands[l] = viterbi_lattice[_i, _j, l] + log_transprob[l, k]

                    opt_l = np.argmax(cands)
                    viterbi_lattice[i, j, k] = cands[opt_l] + log_emitprob_frame[i, j, k]
                    trace_lattice[i, j, k] = opt_l

        # trace
        i, j = len_x, len_y
        map_hstates = [np.argmax(viterbi_lattice[i, j, :])]

        log_likelihood = viterbi_lattice[len_x, len_y, map_hstates[-1]]

        while (i, j) != (0, 0):
            curr_state = map_hstates[-1]
            di, dj = self._delta_index(curr_state)
            map_hstates.append(trace_lattice[i, j, curr_state])
            i -= di
            j -= dj

        map_hstates.reverse()

        return log_likelihood, np.array(map_hstates[1:])

    def fit(self, xseqs, yseqs, max_iter=1000):
        log_transprob = log_(self._transprob)
        log_initprob = log_(self._initprob)

        ## is there better way to explain 0 in log space?(-inf?)

        for i in range(1, max_iter + 1):
            print("{}-th iteration...".format(i))

            # TODO: repalce with generator of seqs
            # TODO: maybe we should hold binary array to explain if the element is zero (because log cannot explain zero exactly)
            # ... it actually could but be careful of overflow of MINF

            accum_initprob = np.zeros_like(log_initprob)
            accum_transprob = np.zeros_like(log_transprob)
            accum_emit = np.zeros_like(self._emitprob)
            accum_gamma = np.zeros_like(log_initprob)

            for j in range(min(xseqs.shape[0], yseqs.shape[0])):
                log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])

                ll, fwd_lattice = self._forward(log_emitprob_frame, log_transprob, log_initprob)
                bwd_lattice = self._backward(log_emitprob_frame, log_transprob)

                gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, ll,
                                                                     log_emitprob_frame, log_transprob)

                shape_x, shape_y, n_hstates = gamma.shape

                for k in range(n_hstates):
                    di, dj = self._delta_index(k)
                    accum_initprob[k] += gamma[di, dj, k]
                    # or just assure zero-value in the last row/colomns and sum over all?
                    accum_transprob[:, k] += np.sum(xi[:shape_x - di, :shape_y - dj, :, k], axis=(0, 1))
                    accum_gamma += np.sum(gamma[:shape_x - di, :shape_y - dj, k], axis=(0,1))
                    emitprob_frame = np.exp(log_emitprob_frame)
                    # TODO: fix emitprob update
                    # how? gamma_value.groub_by(correspoindng_emission(x_i, y_j))
                    accum_emit += np.sum(gamma[:shape_x - di, :shape_y - dj, k] * emitprob_frame[:shape_x - di, :shape_y - dj, k])

            self._initprob = accum_initprob / (np.sum(accum_initprob) + EPS)
            self._transprob = accum_transprob / (np.sum(accum_transprob, axis=1)[:, np.newaxis] + EPS)
            self._emitprob = self._emitprob

        return self

    def _compute_smoothed_marginals(self, fwd_lattice, bwd_lattice, ll,
                                    log_emitprob_frame, log_transprob):
        # smoothed marginal
        shape_x, shape_y, n_hstates = fwd_lattice.shape
        log_gamma = fwd_lattice + bwd_lattice - ll

        # two-sliced smoothed margninal
        log_xi = np.ones((shape_x, shape_y, n_hstates, n_hstates)) * MINF
        for k in range(self._n_hstates):
            di, dj = self._delta_index(k)

            a = fwd_lattice[:shape_x - di, :shape_y - dj, :, np.newaxis] +\
                log_emitprob_frame[di:, dj:, np.newaxis, :] + \
                log_transprob[np.newaxis, np.newaxis, :, :] + \
                bwd_lattice[di:, dj:, np.newaxis, :] - ll

            log_xi[:shape_x - di, :shape_y - dj, :, k] = a[:, :, :, k]

        return np.exp(log_gamma), np.exp(log_xi)

    def _forward(self, log_emitprob_frame, log_transprob, log_initprob):
        shape_x, shape_y, n_hstates = log_emitprob_frame.shape

        fwd_lattice = np.ones((shape_x, shape_y, n_hstates)) * MINF / 10

        for k in range(self._n_hstates):
            di, dj = self._delta_index(k)
            fwd_lattice[di, dj, k] = log_emitprob_frame[di, dj, k] + log_initprob[k]

        for i in range(shape_x):
            for j in range(shape_y):
                if i == 0 and j == 0:
                    continue

                for k in range(n_hstates):
                    di, dj = self._delta_index(k)
                    if i == di and j == dj:
                        continue

                    _i , _j = i - di, j - dj

                    if _i >= 0 and _j >= 0:
                        wbuf = np.ones(n_hstates) * MINF
                        for l in range(n_hstates):
                            wbuf[l] = fwd_lattice[_i, _j, l] + log_transprob[l, k]
                        fwd_lattice[i, j, k] = log_emitprob_frame[i, j, k] + logsumexp(wbuf)


        log_likelihood = logsumexp(fwd_lattice[shape_x - 1, shape_y - 1, :])

        return log_likelihood, fwd_lattice

    def _backward(self, log_emitprob_frame, log_transprob):
        shape_x, shape_y, n_hstates = log_emitprob_frame.shape

        # devide by 10 to avoid overflow when bwd + fwd
        bwd_lattice = np.ones((shape_x, shape_y, n_hstates)) * MINF / 10

        bwd_lattice[shape_x-1, shape_y-1, :] = 0

        for i in reversed(range(shape_x)):
            for j in reversed(range(shape_y)):

                if i == shape_x - 1 and j == shape_y - 1:
                    continue

                for k in range(n_hstates):
                    wbuf = np.ones(n_hstates) * MINF

                    for l in range(n_hstates):
                        di, dj = self._delta_index(l)
                        _i, _j = i + di, j + dj

                        if _i < shape_x and _j < shape_y:
                            wbuf[l] = bwd_lattice[_i, _j, l] + log_transprob[k, l] + log_emitprob_frame[_i, _j, l]

                    bwd_lattice[i, j, k] = logsumexp(wbuf)

        return bwd_lattice

