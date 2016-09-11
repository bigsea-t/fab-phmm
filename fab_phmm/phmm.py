import numpy as np
from fab_phmm.utils import EPS, log_, logsumexp
from fab_phmm import phmmc


# TODO: check if initial probs are valid
# TODO: emit prob matrix for insertion states are redundant (array is sufficient)

class PHMM:

    def __init__(self, n_match_states = 1, n_xins_states=2, n_yins_states=2, n_simbols=4,
                 initprob=None, transprob=None, emitprob=None, stop_threshold=1e-2,
                 link_hstates=False):
        self._initprob = initprob # [n_hstates]
        self._transprob = transprob # [n_hstates, n_hstates]
        self._emitprob = emitprob # [n_hstates, xdim, ydim] (usually xdim == ydim)

        self._n_hstates = n_match_states + n_xins_states + n_yins_states
        self._n_match_states = n_match_states
        self._n_xins_states = n_xins_states
        self._n_yins_states = n_yins_states

        self._hstate_properties = self._gen_hstate_properties()
        self._delta_index_list = [(1, 1), (1, 0), (0, 1)]

        self._n_simbols = n_simbols

        self._stop_threshold = stop_threshold

        self._link_hstates = link_hstates

        if initprob is None or transprob is None or emitprob is None:
            self._params_valid = False
        else:
            self._params_valid = True

    def _params_random_init(self):
        print("params random init")
        initprob = np.random.rand(self._n_hstates)
        initprob /= np.sum(initprob)
        self._initprob = initprob

        transprob = np.random.rand(self._n_hstates, self._n_hstates)

        if not self._link_hstates:
            for j in range(self._n_match_states, self._n_hstates):
                for k in range(j+1, self._n_hstates):
                    transprob[j, k] = 0
                    transprob[k, j] = 0

        transprob /= np.sum(transprob, axis=1)[:, np.newaxis]
        self._transprob = transprob

        self._emitprob = np.zeros((self._n_hstates, self._n_simbols, self._n_simbols))

        for k in range(self._n_hstates):
            if self._hstate_properties[k] == 0:
                emitprob = np.random.rand(self._n_simbols, self._n_simbols)
                emitprob /= np.sum(emitprob)
                self._emitprob[k] = emitprob

            if self._hstate_properties[k] == 1:
                emitprob = np.random.rand(self._n_simbols)
                emitprob /= np.sum(emitprob)
                self._emitprob[k] = np.ones((self._n_simbols, self._n_simbols)) * emitprob[:, np.newaxis]

            if self._hstate_properties[k] == 2:
                emitprob = np.random.rand(self._n_simbols)
                emitprob /= np.sum(emitprob)
                self._emitprob[k] = np.ones((self._n_simbols, self._n_simbols)) * emitprob[np.newaxis, :]

    def _gen_hstate_properties(self):
        # 0: Match, 1: Xins, 2: Yins
        hstate_properties = []
        for _ in range(self._n_match_states):
            hstate_properties.append(0)
        for _ in range(self._n_xins_states):
            hstate_properties.append(1)
        for _ in range(self._n_yins_states):
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
        log_emitprob = log_(self._emitprob)
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

    def score(self, xseq, yseq):
        log_emitprob_frame = self._gen_log_emitprob_frame(xseq, yseq)

        ll, _ = self._forward(log_emitprob_frame, log_(self._transprob), log_(self._initprob))

        return ll

    def _gen_sample_given_hstate(self, hstate):
        if self._hstate_properties[hstate] == 0:
            emitprob_cdf = np.cumsum(self._emitprob[hstate])
            x, y = np.unravel_index((emitprob_cdf > np.random.rand()).argmax(), self._emitprob[hstate].shape)
            return x, y

        if self._hstate_properties[hstate] == 1:
            emitprob_cdf = np.cumsum(self._emitprob[hstate, :, 0])
            x = (emitprob_cdf > np.random.rand()).argmax()
            return x, -1

        if self._hstate_properties[hstate] == 2:
            emitprob_cdf = np.cumsum(self._emitprob[hstate, 0, :])
            y = (emitprob_cdf > np.random.rand()).argmax()
            return -1, y

        raise ValueError("hstate {} is invalid".format(hstate))

    def sample(self, n_samples=1):
        # TODO: random seed
        if not self._params_valid:
            # TODO: should replace as not-fitted error
            raise ValueError("model params are not fitted")

        initprob_cdf = np.cumsum(self._initprob)
        transprob_cdf = np.cumsum(self._transprob, axis=1)

        curr_hstate = (initprob_cdf > np.random.rand()).argmax()
        hseq = [curr_hstate]

        x, y = self._gen_sample_given_hstate(curr_hstate)

        xseq = [x]
        yseq = [y]

        for i in range(1, n_samples):
            curr_hstate = (transprob_cdf[curr_hstate] > np.random.rand()).argmax()
            hseq.append(curr_hstate)
            x, y = self._gen_sample_given_hstate(curr_hstate)
            xseq.append(x)
            yseq.append(y)

        return np.array(xseq), np.array(yseq), np.array(hseq)

    def decode(self, x_seq, y_seq):
        log_initprob = np.log(self._initprob + EPS)
        log_transprob = np.log(self._transprob + EPS)

        len_x = x_seq.shape[0]
        len_y = y_seq.shape[0]

        lattice_shape = (len_x + 1, len_y + 1, self._n_hstates)
        viterbi_lattice = np.ones(lattice_shape) * (- np.inf)
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
                    if viterbi_lattice[i, j, k] != (- np.inf):
                        continue

                    cands = np.ones(self._n_hstates) * (- np.inf)
                    di, dj = self._delta_index(k)
                    _i , _j = i - di, j - dj
                    if _i >= 0 and _j >= 0:
                        for l in range(self._n_hstates):
                            cands[l] = viterbi_lattice[_i, _j, l] + log_transprob[l, k]

                    opt_l = np.argmax(cands)
                    viterbi_lattice[i, j, k] = cands[opt_l] + log_emitprob_frame[i, j, k]
                    trace_lattice[i, j, k] = opt_l

        # trace back
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

    def _print_states(self, ll=None, i_iter=None):
        if i_iter is not None:
            print("{}th iter".format(i_iter))
        if ll is not None:
            print("ll")
            print(ll)
        print("n_hstates: {}".format(self._n_hstates))
        print("n_match_states: {}".format(self._n_match_states))
        print("n_xins_states: {}".format(self._n_xins_states))
        print("n_yins_states: {}".format(self._n_yins_states))

        print("trans")
        print(self._transprob)
        print("init")
        print(self._initprob)
        print("emit")
        print(self._emitprob)
        print()

    def fit(self, xseqs, yseqs, max_iter=1000, verbose=False):

        if not self._params_valid:
            self._params_random_init()

        log_transprob = log_(self._transprob)
        log_initprob = log_(self._initprob)
        assert(len(xseqs) == len(yseqs))
        # is there better way to explain 0 in log space?(-inf?)

        ll_all = - np.inf
        for i in range(1, max_iter + 1):
            ll_all_prev = ll_all
            ll_all = 0

            sstats = self._init_sufficient_statistics()

            for j in range(len(xseqs)):
                log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])

                ll, fwd_lattice = self._forward(log_emitprob_frame, log_transprob, log_initprob)
                ll_all += ll
                bwd_lattice = self._backward(log_emitprob_frame, log_transprob)

                gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, ll,
                                                             log_emitprob_frame, log_transprob)

                self._accumulate_sufficient_statistics(sstats, gamma, xi, xseqs[j], yseqs[j])

            if verbose:
                self._print_states(ll=ll, i_iter=i)

            if (ll_all - ll_all_prev) / len(xseqs) < self._stop_threshold:
                if ll_all - ll_all_prev < 0:
                    raise RuntimeError("log-likelihood decreased")
                else:
                    return ll_all

            self._update_params(sstats)

        self._params_valid = True

        return ll_all

    def _init_sufficient_statistics(self):
        sstats = {}
        sstats["init"] = np.zeros_like(self._initprob)
        sstats["trans"] = np.zeros_like(self._transprob)
        sstats["emit"] = np.zeros_like(self._emitprob)
        return sstats

    def _accumulate_sufficient_statistics(self, sstats, gamma, xi, xseq, yseq):

        shape_x, shape_y, n_hstates = gamma.shape

        for k in range(self._n_hstates):
            di, dj = self._delta_index(k)
            sstats["init"][k] += gamma[di, dj, k]
            # or just assure zero-value in the last row/colomns and sum over all?
            sstats["trans"][:, k] += np.sum(xi[:shape_x - di, :shape_y - dj, :, k], axis=(0, 1))

            if self._hstate_properties[k] == 0:
                for t, x in enumerate(xseq):
                    for u, y in enumerate(yseq):
                        sstats["emit"][k, x, y] += gamma[t + 1, u + 1, k]

            if self._hstate_properties[k] == 1:
                for t, x in enumerate(xseq):
                    for u in range(yseq.shape[0] + 1):
                        sstats["emit"][k, x, :] += gamma[t + 1, u, k]

            if self._hstate_properties[k] == 2:
                for t in range(xseq.shape[0] + 1):
                    for u, y in enumerate(yseq):
                        sstats["emit"][k, :, y] += gamma[t, u + 1, k]

    def _update_params(self, sstats):
        self._initprob = sstats["init"] / (np.sum(sstats["init"]) + EPS)
        self._transprob = sstats["trans"] / (np.sum(sstats["trans"], axis=1)[:, np.newaxis] + EPS)

        for k in range(self._n_hstates):

            if self._hstate_properties[k] == 0:
                self._emitprob[k] = sstats["emit"][k] / (np.sum(sstats["emit"][k]) + EPS)

            if self._hstate_properties[k] == 1:
                self._emitprob[k] = sstats["emit"][k] / (np.sum(sstats["emit"][k], axis=0) + EPS)[np.newaxis, :]

            if self._hstate_properties[k] == 2:
                self._emitprob[k] = sstats["emit"][k] / (np.sum(sstats["emit"][k], axis=1) + EPS)[:, np.newaxis]

    def _compute_smoothed_marginals(self, fwd_lattice, bwd_lattice, ll,
                                    log_emitprob_frame, log_transprob):
        # smoothed marginal
        shape_x, shape_y, n_hstates = fwd_lattice.shape
        log_gamma = fwd_lattice + bwd_lattice - ll

        # two-sliced smoothed margninal
        log_xi = np.ones((shape_x, shape_y, n_hstates, n_hstates)) * (-np.inf)
        for k in range(self._n_hstates):
            di, dj = self._delta_index(k)

            a = fwd_lattice[:shape_x - di, :shape_y - dj, :] + \
                log_emitprob_frame[di:, dj:, np.newaxis, k] + \
                log_transprob[np.newaxis, np.newaxis, :, k] + \
                bwd_lattice[di:, dj:, np.newaxis, k] - ll

            log_xi[:shape_x - di, :shape_y - dj, :, k] = a
        gamma = np.exp(log_gamma)
        xi = np.exp(log_xi)

        return gamma, xi

    def _forward(self, log_emitprob_frame, log_transprob, log_initprob):
        shape_x, shape_y, n_hstates = log_emitprob_frame.shape

        fwd_lattice = np.ones((shape_x, shape_y, n_hstates)) * (- np.inf)

        phmmc._forward(shape_x, shape_y, n_hstates, np.array(self._hstate_properties, dtype=np.int32),
                       log_emitprob_frame, log_initprob, log_transprob, fwd_lattice)

        log_likelihood = logsumexp(fwd_lattice[shape_x - 1, shape_y - 1, :])

        return log_likelihood, fwd_lattice

    def _backward(self, log_emitprob_frame, log_transprob):
        shape_x, shape_y, n_hstates = log_emitprob_frame.shape

        bwd_lattice = np.ones((shape_x, shape_y, n_hstates)) * (-np.inf)

        phmmc._backward(shape_x, shape_y, n_hstates, np.array(self._hstate_properties, dtype=np.int32),
                       log_emitprob_frame, log_transprob, bwd_lattice)

        return bwd_lattice