import numpy as np

EPS = np.finfo(np.float).eps
INF = np.finfo(np.float).max
MINF = np.finfo(np.float).min


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
            params_valid = False

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
        return self._delta_index_list[self._hstate_properties[hstate]]

    def _gen_log_emitprob_frame(self, x_seq, y_seq):
        len_x = x_seq.shape[0]
        len_y = y_seq.shape[0]

        shape = (len_x, len_y, self._n_hstates)
        log_emitprob = np.log(self._emitprob)
        log_emitprob_frame = np.zeros(shape)

        for i in range(len_x):
            for j in range(len_y):
                for k in range(self._n_hstates):
                    log_emitprob_frame[i, j, k] = log_emitprob[k, x_seq[i], y_seq[j]]

        return log_emitprob_frame

    def decode(self, x_seq, y_seq):
        log_initprob = np.log(self._initprob + EPS)
        log_transprob = np.log(self._transprob + EPS)

        len_x = x_seq.shape[0]
        len_y = y_seq.shape[0]

        lattice_shape = (len_x, len_y, self._n_hstates)
        viterbi_lattice = np.zeros(lattice_shape)
        trace_lattice = np.ones(lattice_shape, dtype=np.int32) * (- 1)

        log_emitprob_frame = self._gen_log_emitprob_frame(x_seq, y_seq)

        viterbi_lattice[0, 0, :] = log_emitprob_frame[0, 0, :] + log_initprob

        # induction
        for i in range(len_x):
            for j in range(len_y):

                if i == 0 and j == 0:
                    continue

                for k in range(self._n_hstates):
                    cands = np.ones(self._n_hstates) * MINF

                    for l in range(self._n_hstates):
                        dij = self._delta_index(l)
                        _i , _j = i - dij[0], j - dij[1]
                        if _i >= 0 and _j >= 0:
                            cands[l] = viterbi_lattice[_i, _j, l] + log_transprob[l, k]

                    opt_l = np.argmax(cands)
                    viterbi_lattice[i, j, k] = cands[opt_l] + log_emitprob_frame[i, j, k]
                    trace_lattice[i, j, k] = opt_l

        # trace
        map_hstates = [np.argmax(viterbi_lattice[len_x - 1, len_y - 1, :])]

        log_likelihood = viterbi_lattice[len_x - 1, len_y - 1, map_hstates[-1]]

        i, j = len_x - 1, len_y - 1

        while (i, j) != (0, 0):
            prev_hstate = trace_lattice[i, j, map_hstates[-1]]
            map_hstates.append(prev_hstate)
            dij = self._delta_index(prev_hstate)
            i -= dij[0]
            j -= dij[1]

        map_hstates.reverse()

        return log_likelihood, np.array(map_hstates)

    def fit(self, xseqs, yseqs, max_iter=1000):

        log_transprob = np.log(self._transprob + EPS)
        log_initprob = np.log(self._initprob + EPS)

        for i in range(1, max_iter):
            print("{}-th iteration...".format(i))

            # TODO: repalce with generator of seqs
            for j in range(min(xseqs.shape[0], yseqs.shape[0])):
                log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])

                fwd_lattice = self._forward(log_emitprob_frame, log_transprob, log_initprob)
                bwd_lattice = self._backward(log_emitprob_frame, log_transprob)

        return self

    def _forward(self, log_emitprob_frame, log_transprob, log_initprob):
        len_x, len_y, n_hstates = log_emitprob_frame.shape

        fwd_lattice = np.zeros((len_x, len_y, n_hstates))
        fwd_lattice[0, 0, :] = log_emitprob_frame[0, 0, :] + log_initprob

        for i in range(len_x):
            for j in range(len_y):

                if i == 0 and j == 0:
                    continue

                for k in range(n_hstates):
                    for l in range(n_hstates):
                        dij = self._delta_index(l)
                        _i , _j = i - dij[0], j - dij[1]
                        if _i >= 0 and _j >= 0:
                            fwd_lattice[i, j, k] += fwd_lattice[_i, _j, l] + log_transprob[l, k]

    def _backward(self, log_emitprob_frame, log_transprob):
        len_x, len_y, n_hstates = log_emitprob_frame.shape

        bwd_lattice = np.zeros((len_x, len_y, n_hstates))

        for i in reversed(range(len_x)):
            for j in reversed(range(len_y)):

                if i == len_x - 1 and j == len_y - 1:
                    continue

                for k in range(n_hstates):
                    dij = self._delta_index(k)
                    _i, _j = i + dij[0], j + dij[1]

                    for l in range(n_hstates):
                        if _i < len_x and _j < len_y:
                            bwd_lattice[i, j, k] += bwd_lattice[_i, _j, l] + log_transprob[l, k]

