import numpy as np
from fab_aligners.utils import EPS, log_, logsumexp
from fab_aligners import phmmc
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings

warnings.simplefilter("always")

# TODO: check if initial probs are valid
# TODO: emit prob matrix for insertion states are redundant (array is sufficient)
# TODO: print to file


class PHMM:

    def __init__(self, n_match_states = 1, n_xins_states=2, n_yins_states=2, n_simbols=4,
                 initprob=None, transprob=None, emitprob=None, stop_threshold=1e-2,
                 link_hstates=False, link_match=True):
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
        self._link_match = link_match

        self._last_score = None

        if initprob is None or transprob is None or emitprob is None:
            self._params_valid = False
        else:
            self._params_valid = True

    def _params_random_init(self):
        initprob = np.random.rand(self._n_hstates)
        initprob /= np.sum(initprob)
        self._initprob = initprob

        transprob = np.random.rand(self._n_hstates, self._n_hstates)

        if not self._link_hstates:
            for j in range(self._n_match_states, self._n_hstates):
                for k in range(j+1, self._n_hstates):
                    transprob[j, k] = 0
                    transprob[k, j] = 0

        if not self._link_match:
            for j in range(self._n_match_states):
                for k in range(j+1, self._n_match_states):
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

    def _check_probs(self):
        np.testing.assert_allclose(np.sum(self._initprob), 1)
        np.testing.assert_allclose(np.sum(self._transprob, axis=1), np.ones(self._n_hstates))
        for k in range(self._n_hstates):
            if self._hstate_properties[k] == 0:
                np.testing.assert_allclose(np.sum(self._emitprob[k]), 1)
            elif self._hstate_properties[k] == 1:
                np.testing.assert_allclose(np.sum(self._emitprob[k], axis=0), np.ones(self._n_simbols))
            else:
                np.testing.assert_allclose(np.sum(self._emitprob[k], axis=1), np.ones(self._n_simbols))


    def _gen_hstate_properties(self):
        # 0: Match, 1: Xins, 2: Yins
        hstate_properties = []
        for _ in range(self._n_match_states):
            hstate_properties.append(0)
        for _ in range(self._n_xins_states):
            hstate_properties.append(1)
        for _ in range(self._n_yins_states):
            hstate_properties.append(2)

        return np.array(hstate_properties, dtype=np.int_)

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

        phmmc._compute_log_emitprob_frame(log_emitprob_frame, log_emitprob,
                                          xseq, yseq, self._hstate_properties,
                                          len_x, len_y, self._n_hstates)

        return log_emitprob_frame

    def score(self, xseqs, yseqs):
        ll_all = 0
        log_transprob = log_(self._transprob)
        log_initprob = log_(self._initprob)
        assert(len(xseqs) == len(yseqs))
        for j in range(len(xseqs)):
            log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])

            ll, _ = self._forward(log_emitprob_frame, log_transprob, log_initprob)

            ll_all += ll

        return ll_all

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
        assert(x_seq.shape[0] > 0 and y_seq.shape[0] > 0)
        # print("emit")
        log_initprob = log_(self._initprob)
        log_transprob = log_(self._transprob)

        len_x = x_seq.shape[0]
        len_y = y_seq.shape[0]

        lattice_shape = (len_x + 1, len_y + 1, self._n_hstates)
        viterbi_lattice = np.full(lattice_shape, -np.inf)

        log_emitprob_frame = self._gen_log_emitprob_frame(x_seq, y_seq)

        map_hstates = np.full(len_x + len_y + 1, -1, dtype=np.int_)
        ll = phmmc._decode(len_x+1, len_y+1, self._n_hstates,
                          self._hstate_properties, log_emitprob_frame,
                          log_initprob, log_transprob,
                          viterbi_lattice, map_hstates)

        l = map_hstates.shape[0]
        map_hstates = map_hstates[:l-(map_hstates == -1).sum()]

        return ll, map_hstates

    def decode_align(self, xseq, yseq):
        _, map_hstates = self.decode(xseq, yseq)

        aligned_xseq = []
        aligned_yseq = []

        xseq = list(xseq)
        yseq = list(yseq)
        for h in map_hstates:
            prop = self._hstate_properties[h]
            if prop == 0:
                aligned_xseq.append(xseq.pop(0))
                aligned_yseq.append(yseq.pop(0))
            elif prop == 1:
                aligned_xseq.append(xseq.pop(0))
                aligned_yseq.append(-1)
            elif prop == 2:
                aligned_xseq.append(-1)
                aligned_yseq.append(yseq.pop(0))

        assert len(xseq) == 0
        assert len(yseq) == 0

        return np.array(aligned_xseq), np.array(aligned_yseq)

    def gen_alignment_matrix(self, xseq, yseq):
        _, map_hstates = self.decode(xseq, yseq)
        xlen, = xseq.shape
        ylen, = yseq.shape
        matrix = np.zeros((xlen+1, ylen+1), dtype=np.int_)

        i, j = (0, 0)

        for h in map_hstates:
            di, dj = self._delta_index(h)
            i += di
            j += dj
            matrix[i, j] = 1
        return matrix

    def _print_states(self, i_iter=None, verbose_level=1):

        if verbose_level >= 1:
            if i_iter is not None:
                print("{}th iter".format(i_iter))
            print("score")
            print(self._last_score)
            print("n_hstates: {}".format(self._n_hstates))
            print("n_match_states: {}".format(self._n_match_states))
            print("n_xins_states: {}".format(self._n_xins_states))
            print("n_yins_states: {}".format(self._n_yins_states))

        if verbose_level >= 2:
            print("trans")
            print(self._transprob)
            print("init")
            print(self._initprob)
            print("emit")
            print(self._emitprob)

        if verbose_level >= 1:
            print()

        sys.stdout.flush()

    def _compute_sstats(self, sstats, xseq, yseq, log_transprob, log_initprob, lock):
        log_emitprob_frame = self._gen_log_emitprob_frame(xseq, yseq)

        ll, fwd_lattice = self._forward(log_emitprob_frame, log_transprob, log_initprob)
        bwd_lattice = self._backward(log_emitprob_frame, log_transprob)

        gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, ll,
                                                     log_emitprob_frame, log_transprob)

        with lock:
            self._accumulate_sufficient_statistics(sstats, ll, gamma, xi, xseq, yseq)


    def fit(self, xseqs, yseqs, max_iter=1000, verbose=False, verbose_level=1, n_threads=4):

        if not self._params_valid:
            self._params_random_init()

        assert(len(xseqs) == len(yseqs))
        # is there better way to explain 0 in log space?(-inf?)

        if self._last_score is None:
            self._last_score = - np.inf

        for i in range(1, max_iter + 1):
            sstats = self._init_sufficient_statistics()

            log_transprob = log_(self._transprob)
            log_initprob = log_(self._initprob)

            lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=n_threads) as e:
                for j in range(len(xseqs)):
                    e.submit(self._compute_sstats,
                             sstats, xseqs[j], yseqs[j],
                             log_transprob, log_initprob, lock)
                
            if verbose:
                self._print_states(i_iter=i, verbose_level=verbose_level)

            if np.abs(sstats["score"] - self._last_score) / len(xseqs) < self._stop_threshold:
                return self._last_score
            elif sstats["score"] - self._last_score < 0:
                warnings.warn("score decreased", RuntimeWarning)
                return
                #raise RuntimeError("log-likelihood decreased")

            self._update_params(sstats)

        self._params_valid = True

        return self._last_score

    def _init_sufficient_statistics(self):
        sstats = {}
        sstats["init"] = np.zeros_like(self._initprob)
        sstats["trans"] = np.zeros_like(self._transprob)
        sstats["emit"] = np.zeros_like(self._emitprob)
        sstats["score"] = 0
        return sstats

    def _accumulate_sufficient_statistics(self, sstats, ll, gamma, xi, xseq, yseq):

        shape_x, shape_y, n_hstates = gamma.shape

        sstats["score"] += ll

        delta_list = [(0, 0), (0, 1), (1, 0)]
        for k in range(self._n_hstates):
            di, dj = self._delta_index(k)
            sstats["init"][k] += gamma[di, dj, k]
            # or just assure zero-value in the last row/colomns and sum over all?
            sstats["trans"][:, k] += np.sum(xi[:shape_x - di, :shape_y - dj, :, k], axis=(0, 1))

            if self._hstate_properties[k] == 0:
                phmmc._accumulate_emitprob_match(sstats["emit"], gamma,
                                                 xseq, yseq, xseq.shape[0], yseq.shape[0], k)

            if self._hstate_properties[k] == 1:
                phmmc._accumulate_emitprob_xins(sstats["emit"], gamma,
                                                xseq, yseq, xseq.shape[0], yseq.shape[0], k)

            if self._hstate_properties[k] == 2:
                phmmc._accumulate_emitprob_yins(sstats["emit"], gamma,
                                                xseq, yseq, xseq.shape[0], yseq.shape[0], k)

    def _update_params(self, sstats):
        self._last_score = sstats["score"]
        self._initprob = sstats["init"] / (np.sum(sstats["init"]))

        self._transprob = (sstats["trans"]) / (np.sum(sstats["trans"], axis=1)[:, np.newaxis])

        for k in range(self._n_hstates):

            if self._hstate_properties[k] == 0:
                self._emitprob[k] = sstats["emit"][k] / (np.sum(sstats["emit"][k]))

            if self._hstate_properties[k] == 1:
                self._emitprob[k] = sstats["emit"][k] / (np.sum(sstats["emit"][k], axis=0))[np.newaxis, :]

            if self._hstate_properties[k] == 2:
                self._emitprob[k] = sstats["emit"][k] / (np.sum(sstats["emit"][k], axis=1))[:, np.newaxis]

    def _compute_smoothed_marginals(self, fwd_lattice, bwd_lattice, ll,
                                    log_emitprob_frame, log_transprob):
        # smoothed marginal
        shape_x, shape_y, n_hstates = fwd_lattice.shape
        log_gamma = fwd_lattice + bwd_lattice - ll
        gamma = np.exp(log_gamma)

        # two-sliced smoothed margninal
        log_xi = np.full((shape_x, shape_y, n_hstates, n_hstates), -np.inf)
        phmmc._compute_two_sliced_margnial(shape_x, shape_y, n_hstates,
                                           self._hstate_properties,
                                           ll, fwd_lattice, bwd_lattice, log_emitprob_frame,
                                           log_transprob, log_xi)
        xi = np.exp(log_xi)

        return gamma, xi

    def _forward(self, log_emitprob_frame, log_transprob, log_initprob):
        shape_x, shape_y, n_hstates = log_emitprob_frame.shape

        fwd_lattice = np.full((shape_x, shape_y, n_hstates), -np.inf)

        phmmc._forward(shape_x, shape_y, n_hstates, self._hstate_properties,
                       log_emitprob_frame, log_initprob, log_transprob, fwd_lattice)

        log_likelihood = logsumexp(fwd_lattice[shape_x - 1, shape_y - 1, :])

        return log_likelihood, fwd_lattice

    def _backward(self, log_emitprob_frame, log_transprob):
        shape_x, shape_y, n_hstates = log_emitprob_frame.shape

        bwd_lattice = np.full((shape_x, shape_y, n_hstates), - np.inf)

        phmmc._backward(shape_x, shape_y, n_hstates, self._hstate_properties,
                       log_emitprob_frame, log_transprob, bwd_lattice)

        return bwd_lattice