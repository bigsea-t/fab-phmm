from fab_aligners.phmm import PHMM
from fab_aligners.utils import *
import warnings
import copy
import timeit
from concurrent.futures import ThreadPoolExecutor
import threading
import sys

warnings.simplefilter("always")

class Monitor:
    def __init__(self, threshold=1e-5):
        self.records = []
        self.threshold = threshold

    def write(self, score, n_hstates, sstats):
        record = {}
        record["time"] = timeit.default_timer()
        record["score"] = score
        record["n_hstates"] = n_hstates
        record["sstats"] = sstats
        self.records.append(record)

    def last_shrinked(self):
        if len(self.records) < 2:
            return False

        old = self.records[-2]["n_hstates"]
        new = self.records[-1]["n_hstates"]

        return new != old

    def is_converged(self):
        if len(self.records) < 2:
            return False

        old = self.records[-2]["score"]
        new = self.records[-1]["score"]

        diff = (new - old)

        if np.isinf(new):
            warnings.warn("score diverge to infinity", RuntimeWarning)
            return True

        if np.abs(diff) < self.threshold:
            return True
        elif diff < 0 and not self.last_shrinked():
            warnings.warn("score decreased", RuntimeWarning)
            return False
        else:
            return False

def _incremental_search(xseqs, yseqs, n_match, n_xins, n_yins,
                        stop_threshold=1e-5, shrink_threshold=1e-2,
                        max_iter=1000, verbose=False,
                        verbose_level=1, max_match_states=10, max_ins_states=10, visited=None, n_threads=4):
    if visited is None:
        visited = set()

    if (n_match, n_xins, n_yins) in visited:
        return []
    visited.add((n_match, n_xins, n_yins))

    if n_match > max_match_states or n_xins > max_ins_states or n_yins > max_ins_states:
        return []

    model = FABPHMM(n_match_states=n_match,
                    n_xins_states=n_xins,
                    n_yins_states=n_yins,
                    shrink_threshold=shrink_threshold,
                    stop_threshold=stop_threshold,
                    shrink=False)
    model.fit(xseqs, yseqs,
              max_iter=max_iter, verbose=verbose, verbose_level=verbose_level,
              n_threads=n_threads)

    if verbose:
        model._print_states()

    if model._last_shrinked:
        print("=== shrinked: no more serach====")
        return []

    models = [model]

    deltas = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for d in deltas:
        models += _incremental_search(xseqs, yseqs, n_match + d[0], n_xins + d[1], n_yins + d[2],
                                      stop_threshold=stop_threshold, max_iter=max_iter, shrink_threshold=shrink_threshold,
                                      verbose=verbose, verbose_level=verbose_level, max_match_states=max_match_states, max_ins_states=max_ins_states,
                                      visited=visited, n_threads=n_threads)

    return models


def incremental_model_selection(xseqs, yseqs,
                                stop_threshold=1e-5,
                                shrink_threshold=1e-2,
                                max_iter=1000,
                                verbose=False,
                                verbose_level=1,
                                max_match_states=10,
                                max_ins_states=10,
                                sorted=True,
                                n_threads=4):

    models = _incremental_search(xseqs, yseqs, 1, 1, 1,
                                 stop_threshold=stop_threshold, shrink_threshold=shrink_threshold,
                                 max_iter=max_iter, verbose=verbose, verbose_level=verbose_level,
                                 max_match_states=max_match_states, max_ins_states=max_ins_states, n_threads=n_threads)

    if sorted:
        models.sort(key=lambda m: - m._last_score)

    return models


def decremental_greedy_selection(xseqs, yseqs,
                                 stop_threshold=1e-5,
                                 shrink_threshold=1e-2,
                                 max_iter=1000,
                                 verbose=False,
                                 verbose_level=1,
                                 max_match_states=10,
                                 max_ins_states=5,
                                 sorted=True,
                                 n_threads=2):
    models = []

    model = FABPHMM(n_match_states=max_match_states,
                    n_xins_states=max_ins_states,
                    n_yins_states=max_ins_states,
                    shrink_threshold=shrink_threshold,
                    stop_threshold=stop_threshold,
                    shrink=True)
    while True:
        model.fit(xseqs, yseqs, max_iter=max_iter, verbose=verbose, verbose_level=verbose_level, n_threads=n_threads)
        models.append(copy.deepcopy(model))
        if not model.greedy_shrink(xseqs, yseqs):
            break

    if sorted:
        models.sort(key=lambda m: - m._last_score)

    return models


class FABPHMM(PHMM):

    def __init__(self, n_match_states=1, n_xins_states=2, n_yins_states=2,
                 n_simbols=4, initprob=None, transprob=None, emitprob=None,
                 symmetric_emission=False, shrink_threshold=1e-2,
                 stop_threshold=1e-5, link_hstates=False, shrink=True,
                 propdim_count_nonzero=True, link_match=True):

        super(FABPHMM, self).__init__(n_match_states=n_match_states,
                                      n_xins_states=n_xins_states,
                                      n_yins_states=n_yins_states,
                                      n_simbols=n_simbols,
                                      initprob=initprob,
                                      transprob=transprob,
                                      emitprob=emitprob,
                                      stop_threshold=stop_threshold,
                                      link_hstates=link_hstates,
                                      link_match=link_match)
        # implement symmetric one for the first program

        self._symmetric_emission = symmetric_emission
        self._shrink_threshold = shrink_threshold
        self._shrink = shrink
        self._propdim_count_nonzero = propdim_count_nonzero
        self._qsum_emit = None
        self._qsum_trans = None

        self._done_shrink = None
        self._last_shrinked = False

        self.monitor = None

        if symmetric_emission:
            raise NotImplementedError("Symmetric emission is not implemented")
            # not implemented
            #self._dim_match_emit = n_simbols * (n_simbols + 1) // 2 - 1
            #self._dim_ins_emit = n_simbols // 2 # /?? how can I deal with it??
            # it is wired because emission parameter demension of insertion and deletion
            # should be symmetric but in this case we have to compute dimenson separately
        else:
            self._dim_match_emit = n_simbols ** 2 - 1
            self._dim_ins_emit = n_simbols - 1

    def _gen_dims(self):
        # assume symmetric output prob disribution
        n_hstates = self._n_hstates
        dim_emit = np.zeros(n_hstates)
        dim_trans = np.sum(self._transprob != 0, axis=1) - 1

        dim_init = np.sum(self._initprob != 0) - 1

        for k in range(n_hstates):
            hstate_prop = self._hstate_properties[k]
            dim_emit[k] = self._dim_match_emit if hstate_prop == 0 else self._dim_ins_emit
        return dim_init, dim_trans, dim_emit

    def _gen_logdelta(self, log_emitprob_frame, dims_trans, dims_emit):
        term_trans = - dims_trans / (2 * self._qsum_trans + EPS)
        term_emit = - dims_emit / (2 * self._qsum_emit + EPS)

        logdelta_1 = term_trans + term_emit
        logdelta_2 = term_emit

        xshape, yshape, n_hstates = log_emitprob_frame.shape

        pseudo_prob = np.full((xshape, yshape, n_hstates), logdelta_1)
        pseudo_prob[-1, -1, :] = logdelta_2

        return pseudo_prob

    def _get_state_to_delete(self):
        if self._n_match_states == 1 and self._n_xins_states == 1 and self._n_yins_states == 1:
            return -1, -1

        dim_init, dims_trans, dims_emit = self._gen_dims()
        qscore = self._qsum_emit / (dims_trans + dims_emit)
        qscore /= np.sum(qscore)

        sep_mx = self._n_match_states
        sep_xy = self._n_match_states + self._n_xins_states

        qscore_match = qscore[:sep_mx]
        qscore_xins = qscore[sep_mx:sep_xy]
        qscore_yins = qscore[sep_xy:]

        def min_val_index(qsum, origin):
            argmin = np.argmin(qsum)
            min_val = qsum[argmin]
            min_index = argmin + origin
            return min_val, min_index

        cands = []
        if self._n_match_states > 1:
            cands.append(min_val_index(qscore_match, 0))
        if self._n_xins_states > 1:
            cands.append(min_val_index(qscore_xins, sep_mx))
        if self._n_yins_states > 1:
            cands.append(min_val_index(qscore_yins, sep_xy))

        val, state = min(cands, key=lambda k: k[0])

        return val, state

    def _shrinkage_operation(self, force=False):
        val, state = self._get_state_to_delete()

        if state == -1:
            return 0

        if force:
            self._delete_state(state)
            return 1

        n_deleted = 0
        while val < self._shrink_threshold:
            self._delete_state(state)
            n_deleted += 1
            val, state = self._get_state_to_delete()
            if state == -1:
                break

        return n_deleted

    def _init_sufficient_statistics(self):
        sstats = super(FABPHMM, self)._init_sufficient_statistics()
        sstats["qsum_emit"] = np.zeros(self._n_hstates)
        sstats["qsum_trans"] = np.zeros(self._n_hstates)
        return sstats

    def score(self, xseqs, yseqs, type="fic"):
        if type == "fic":
            raise NotImplemented("type fic is not implemented")
        elif type == "ll":
            return super(FABPHMM, self).score(xseqs, yseqs)
        else:
            raise ValueError("invalid type")

    def _accumulate_sufficient_statistics(self, sstats, free_energy, gamma, xi, xseq, yseq):
        # free energy is accumulated as sstats["score"]
        super(FABPHMM, self)._accumulate_sufficient_statistics(sstats, free_energy, gamma, xi, xseq, yseq)
        sum_gamma = np.sum(gamma, axis=(0, 1))
        sstats["qsum_emit"] += sum_gamma
        sstats["qsum_trans"] += sum_gamma - gamma[-1, -1, :]

    def _gen_random_qsum(self, xseqs, yseqs):
        n_seqs = len(xseqs)
        qsum_emit = np.zeros(self._n_hstates)
        for i in range(n_seqs):
            xlen, ylen = xseqs[i].shape[0], yseqs[i].shape[0]
            frame = np.random.rand(xlen + 1, ylen + 1, self._n_hstates)
            frame /= frame.sum(axis=2)[:, :, np.newaxis]
            qsum_emit += np.sum(frame, axis=(0,1))
            qsum_trans = qsum_emit - frame[-1, -1, :]
        return qsum_emit, qsum_trans

    def _update_params(self, sstats, fic):
        super(FABPHMM, self)._update_params(sstats)
        self._qsum_emit = sstats["qsum_emit"]
        self._qsum_trans = sstats["qsum_trans"]
        self._last_score = fic
        self._check_probs()

    def _calculate_fic(self, sstats, n_seq):
        dim_init, dims_trans, dims_emit = self._gen_dims()

        fic = sstats["score"] \
              - dim_init / 2 * np.log(n_seq) \
              - np.sum(dims_trans / 2 * (np.log(sstats["qsum_trans"]) - 1)) \
              - np.sum(dims_emit / 2 * (np.log(sstats["qsum_emit"]) - 1))
        return fic

    def greedy_shrink(self):
        return self._shrinkage_operation(force=True)

    def _delete_state(self, deleted_state):

        preserved_hstates = np.delete(np.arange(self._n_hstates), deleted_state)

        deleted_hstateprop = self._hstate_properties[deleted_state]

        if deleted_hstateprop == 0:
            self._n_match_states -= 1
        elif deleted_hstateprop == 1:
            self._n_xins_states -= 1
        elif deleted_hstateprop == 2:
            self._n_yins_states -= 1
        else:
            raise ValueError("invalid hstate prop")

        assert(self._n_match_states > 0)
        assert(self._n_xins_states > 0)
        assert(self._n_yins_states > 0)

        self._n_hstates -= 1

        self._hstate_properties = self._gen_hstate_properties()

        self._initprob = self._initprob[preserved_hstates]
        self._initprob /= np.sum(self._initprob)
        self._transprob = self._transprob[:, preserved_hstates]
        self._transprob = self._transprob[preserved_hstates, :]
        self._transprob = self._transprob / np.sum(self._transprob, axis=1)[:, np.newaxis]
        self._emitprob = self._emitprob[preserved_hstates, :, :]
        for k in range(self._n_hstates):
            if self._hstate_properties[k] == 0:
                self._emitprob[k] = self._emitprob[k] / np.sum(self._emitprob[k])

            if self._hstate_properties[k] == 1:
                self._emitprob[k] = self._emitprob[k] / np.sum(self._emitprob[k], axis=0)[np.newaxis, :]

            if self._hstate_properties[k] == 2:
                self._emitprob[k] = self._emitprob[k] / np.sum(self._emitprob[k], axis=1)[:, np.newaxis]

        self._qsum_emit = self._qsum_emit[preserved_hstates] + self._qsum_emit[deleted_state] / self._n_hstates
        self._qsum_trans = self._qsum_trans[preserved_hstates] + self._qsum_trans[deleted_state] / self._n_hstates

        self._check_probs()

        self._last_shrinked = True

    def _compute_sstats(self, sstats, xseq, yseq, dims_trans, dims_emit, log_transprob, log_initprob, lock):
        log_emitprob_frame = self._gen_log_emitprob_frame(xseq, yseq)

        log_delta = self._gen_logdelta(log_emitprob_frame, dims_trans, dims_emit)

        fab_log_emitprob_frame = log_emitprob_frame + log_delta

        free_energy, fwd_lattice = self._forward(fab_log_emitprob_frame, log_transprob, log_initprob)

        bwd_lattice = self._backward(fab_log_emitprob_frame, log_transprob)

        gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, free_energy,
                                                     fab_log_emitprob_frame, log_transprob)

        with lock:
            self._accumulate_sufficient_statistics(sstats, free_energy, gamma, xi, xseq, yseq)

    def fit(self, xseqs, yseqs, max_iter=1000, verbose=False, verbose_level=1, n_threads=4):
        if not self._params_valid:
            self._params_random_init()
            self._params_valid = True

        assert(len(xseqs) == len(yseqs))
        assert(len(xseqs) > 0)
        n_seq = len(xseqs)

        sys.stdout.flush()

        self._qsum_emit, self._qsum_trans = self._gen_random_qsum(xseqs, yseqs)

        if self._last_score is None:
            self._last_score = - np.inf

        if self.monitor is None:
            self.monitor = Monitor(threshold=self._stop_threshold)

        i = 0
        while i < max_iter:
            i += 1
            print("{} th iter".format(i))

            log_transprob = log_(self._transprob)
            log_initprob = log_(self._initprob)

            self._last_shrinked = False
            sstats = self._init_sufficient_statistics()

            dim_init, dims_trans, dims_emit = self._gen_dims()

            lock = threading.Lock()

            if n_threads == 0:
                for j in range(n_seq):
                    self._compute_sstats(sstats, xseqs[j], yseqs[j],
                        dims_trans, dims_emit,
                        log_transprob, log_initprob, lock)
            else:
                with ThreadPoolExecutor(max_workers=n_threads) as e:
                    for j in range(n_seq):
                        e.submit(self._compute_sstats,
                                 sstats, xseqs[j], yseqs[j],
                                 dims_trans, dims_emit,
                                 log_transprob, log_initprob, lock)

            fic = self._calculate_fic(sstats, n_seq)

            self.monitor.write(fic / n_seq, self._n_hstates, sstats)

            if verbose:
                self._print_states(i_iter=i, verbose_level=verbose_level)

            self._update_params(sstats, fic)

            if self.monitor.is_converged():
                print("converged with fic {}".format(fic))
                return self

            # dry is unavailable
            n_shrinked_states = self._shrinkage_operation()

            # TODO: this part should be integrated into Monitor
            if n_shrinked_states > 0:
                i = 0
                if not self._shrink:
                    print("shrinked: not an optimal model")
                    return self

            sys.stdout.flush()

        warnings.warn("end fitting though not yet converged", RuntimeWarning)
        return self
