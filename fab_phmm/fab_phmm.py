from fab_phmm.phmm import PHMM
from fab_phmm.utils import *
import warnings
import sys
import copy
import timeit


class Monitor:
    def __init__(self, threshold=1e-5):
        self.records = []
        self.threshold = threshold

    def write(self, score, n_hstates):
        record = {}
        record["time"] = timeit.default_timer()
        record["score"] = score
        record["n_hstates"] = n_hstates
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

        if np.isinf(new):
            warnings.warn("score diverge to infinity", RuntimeWarning)
            return True

        if np.abs(new - old) < self.threshold:
            return True
        elif new - old < 0 and not self.last_shrinked():
            raise RuntimeError("score decreased")
        else:
            return False


def _incremental_search(xseqs, yseqs, n_match, n_xins, n_yins,
                        stop_threshold=1e-5, shrink_threshold=1e-2,
                        max_iter=1000, verbose=False,
                        verbose_level=1, max_n_states=10, visited=None):
    if visited is None:
        visited = set()

    if (n_match, n_xins, n_yins) in visited:
        return []
    visited.add((n_match, n_xins, n_yins))

    if n_match > max_n_states or n_xins > max_n_states or n_yins > max_n_states:
        return []

    model = FABPHMM(n_match_states=n_match,
                    n_xins_states=n_xins,
                    n_yins_states=n_yins,
                    shrink_threshold=shrink_threshold,
                    stop_threshold=stop_threshold,
                    shrink=False)
    model.fit(xseqs, yseqs,
              max_iter=max_iter, verbose=verbose, verbose_level=verbose_level)

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
                                      verbose=verbose, verbose_level=verbose_level, max_n_states=10,
                                      visited=visited)

    return models


def incremental_model_selection(xseqs, yseqs,
                                stop_threshold=1e-5,
                                shrink_threshold=1e-2,
                                max_iter=1000,
                                verbose=False,
                                verbose_level=1,
                                max_n_states=10,
                                sorted=True):

    models = _incremental_search(xseqs, yseqs, 1, 1, 1,
                                 stop_threshold=stop_threshold, shrink_threshold=shrink_threshold,
                                 max_iter=max_iter, verbose=verbose, verbose_level=verbose_level, max_n_states=max_n_states)

    if sorted:
        models.sort(key=lambda m: - m._last_score)

    return models


def decremental_greedy_selection(xseqs, yseqs,
                                 stop_threshold=1e-5,
                                 shrink_threshold=1e-2,
                                 max_iter=1000,
                                 verbose=False,
                                 verbose_level=1,
                                 max_n_states=10,
                                 sorted=True):
    models = []

    model = FABPHMM(n_match_states=max_n_states,
                    n_xins_states=max_n_states,
                    n_yins_states=max_n_states,
                    shrink_threshold=shrink_threshold,
                    stop_threshold=stop_threshold,
                    shrink=True)
    while True:
        model.fit(xseqs, yseqs, max_iter=max_iter, verbose=verbose, verbose_level=verbose_level)
        models.append(copy.deepcopy(model))
        if not model.greedy_shrink():
            break

    if sorted:
        models.sort(key=lambda m: - m._last_score)

    return models



class FABPHMM(PHMM):

    def __init__(self, n_match_states=1, n_xins_states=2, n_yins_states=2,
                 n_simbols=4, initprob=None, transprob=None, emitprob=None,
                 symmetric_emission=False, shrink_threshold=1e-2,
                 stop_threshold=1e-5, link_hstates=False, shrink=True,
                 propdim_count_nonzero=True):

        super(FABPHMM, self).__init__(n_match_states=n_match_states,
                                      n_xins_states=n_xins_states,
                                      n_yins_states=n_yins_states,
                                      n_simbols=n_simbols,
                                      initprob=initprob,
                                      transprob=transprob,
                                      emitprob=emitprob,
                                      stop_threshold=stop_threshold,
                                      link_hstates=link_hstates)
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

        if self._propdim_count_nonzero:
            dim_trans = np.sum(self._transprob != 0, axis=1) - 1
            dim_init = np.sum(self._initprob != 0) - 1
        else:
            dim_trans = np.ones(n_hstates) * (n_hstates - 1)
            dim_init = n_hstates - 1

        for k in range(n_hstates):
            hstate_prop = self._hstate_properties[k]
            dim_emit[k] = self._dim_match_emit if hstate_prop == 0 else self._dim_ins_emit

        return dim_init, dim_trans, dim_emit


    def _gen_pseudo_prob(self, log_emitprob_frame, dims_trans, dims_emit):
        term_trans = - dims_trans / (2 * self._qsum_trans + EPS)
        term_emit = - dims_emit / (2 * self._qsum_emit + EPS)

        score_1 = np.exp(term_trans + term_emit)
        norm_1 = np.sum(score_1)
        score_2 = np.exp(term_emit)
        norm_2 = np.sum(score_2)

        xshape, yshape, n_hstates = log_emitprob_frame.shape

        norm = np.ones((xshape, yshape)) * norm_1
        norm[-1, -1] = norm_2

        pseudo_prob = np.ones((xshape, yshape, n_hstates)) * (score_1 / norm_1)[np.newaxis, np.newaxis, :]
        pseudo_prob[-1, -1, :] = score_2 / norm_2

        return norm, pseudo_prob

    def _shrinkage_operation(self, dry=False):
        below_threshold = (self._qsum_emit / np.sum(self._qsum_emit)) < (self._shrink_threshold)
        preserved_hstates, = np.nonzero(~below_threshold)
        deleted_hstates, = np.nonzero(below_threshold)

        n_deleted_hstateprops = np.zeros(3, dtype=np.int)
        for k in deleted_hstates:
            n_deleted_hstateprops[self._hstate_properties[k]] += 1

        if np.sum(n_deleted_hstateprops) == 0:
            return 0

        self._last_shrinked = True

        if dry:
            return np.sum(n_deleted_hstateprops)

        self._n_match_states -= n_deleted_hstateprops[0]
        self._n_xins_states -= n_deleted_hstateprops[1]
        self._n_yins_states -= n_deleted_hstateprops[2]
        assert(self._n_match_states > 0)
        assert(self._n_xins_states > 0)
        assert(self._n_yins_states > 0)

        self._n_hstates -= np.sum(n_deleted_hstateprops)

        self._hstate_properties = self._gen_hstate_properties()

        self._initprob = self._initprob[preserved_hstates]
        self._transprob = self._transprob[:, preserved_hstates]
        self._transprob = self._transprob[preserved_hstates, :]
        self._emitprob = self._emitprob[preserved_hstates, :, :]
        self._qsum_emit = self._qsum_emit[preserved_hstates]
        self._qsum_trans = self._qsum_trans[preserved_hstates]

        for k in deleted_hstates:
            print("hstate {} is deleted".format(k))

        return np.sum(n_deleted_hstateprops)

    def _init_sufficient_statistics(self):
        sstats = super(FABPHMM, self)._init_sufficient_statistics()
        sstats["qsum_emit"] = np.zeros(self._n_hstates)
        sstats["qsum_trans"] = np.zeros(self._n_hstates)
        sstats["norm_sum"] = 0
        return sstats

    def score(self, xseqs, yseqs, type="fic"):
        if type == "fic":
            raise NotImplemented("type fic is not implemented")
        elif type == "ll":
            return super(FABPHMM, self).score(xseqs, yseqs)
        else:
            raise ValueError("invalid type")


    def _accumulate_sufficient_statistics(self, sstats, free_energy, gamma, xi, xseq, yseq, norm):
        # free energy is accumulated as sstats["score"]
        super(FABPHMM, self)._accumulate_sufficient_statistics(sstats, free_energy, gamma, xi, xseq, yseq)
        sstats["qsum_emit"] += np.sum(gamma, axis=(0, 1))
        sstats["qsum_trans"] += sstats["qsum_emit"] - gamma[-1, -1, :]
        sstats["norm_sum"] += np.sum(np.log(norm) * np.sum(gamma, axis=2))

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

    def _calculate_fic(self, sstats, n_seq):
        dim_init, dims_trans, dims_emit = self._gen_dims()
        fic = sstats["score"] \
              - dim_init / 2 * np.log(n_seq) \
              - np.sum(dims_trans / 2 * (np.log(sstats["qsum_trans"]) - 1)) \
              - np.sum(dims_emit / 2 * (np.log(sstats["qsum_emit"]) - 1))
        return fic

    def greedy_shrink(self):
        if self._n_match_states == 1 and self._n_xins_states == 1 and self._n_yins_states == 1:
            return False

        sep_mx = self._n_match_states
        sep_xy = self._n_match_states + self._n_xins_states

        qsum_emit_match = self._qsum_emit[:sep_mx]
        qsum_emit_xins = self._qsum_emit[sep_mx:sep_xy]
        qsum_emit_yins = self._qsum_emit[sep_xy:]

        def min_val_index(qsum, origin):
            argmin = np.argmin(qsum)
            min_val = qsum_emit_match[argmin] / np.sum(qsum)
            min_index = argmin + origin
            return min_val, min_index

        cands = [min_val_index(qsum_emit_match, 0),
                 min_val_index(qsum_emit_xins, sep_mx),
                 min_val_index(qsum_emit_yins, sep_xy)]

        _, deleted_state = min(cands, key=lambda k: k[0])
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
        self._transprob = self._transprob[:, preserved_hstates]
        self._transprob = self._transprob[preserved_hstates, :]
        self._emitprob = self._emitprob[preserved_hstates, :, :]
        self._qsum_emit = self._qsum_emit[preserved_hstates]
        self._qsum_trans = self._qsum_trans[preserved_hstates]

        print("hstate {} is deleted".format(deleted_state))

        self._last_shrinked = True

        return True

    def fit(self, xseqs, yseqs, max_iter=1000, verbose=False, verbose_level=1):
        if not self._params_valid:
            self._params_random_init()
            self._params_valid = True

        assert(len(xseqs) == len(yseqs))
        n_seq = len(xseqs)

        self._qsum_emit, self._qsum_trans = self._gen_random_qsum(xseqs, yseqs)

        if self._last_score is None:
            self._last_score = - np.inf

        # TODO: class Recorder / integrate some features into this
        if self.monitor is None:
            self.monitor = Monitor(threshold=self._shrink_threshold)

        for i in range(1, max_iter + 1):
            print("{} th iter".format(i))

            log_transprob = log_(self._transprob)
            log_initprob = log_(self._initprob)

            self._last_shrinked = False
            sstats = self._init_sufficient_statistics()

            dim_init, dims_trans, dims_emit = self._gen_dims()

            for j in range(n_seq):
                log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])

                norm, pseudo_prob = self._gen_pseudo_prob(log_emitprob_frame, dims_trans, dims_emit)
                fab_log_emitprob_frame = log_emitprob_frame + log_(pseudo_prob * norm[:, :, np.newaxis])

                free_energy, fwd_lattice = self._forward(fab_log_emitprob_frame, log_transprob, log_initprob)

                bwd_lattice = self._backward(fab_log_emitprob_frame, log_transprob)

                gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, free_energy,
                                                             fab_log_emitprob_frame, log_transprob)

                self._accumulate_sufficient_statistics(sstats, free_energy, gamma, xi, xseqs[j], yseqs[j], norm)

            fic = self._calculate_fic(sstats, n_seq)

            self.monitor.write(fic, self._n_hstates)

            if verbose:
                self._print_states(i_iter=i, verbose_level=verbose_level)

            if self.monitor.is_converged():
                print("converged with fic {}".format(fic))
                return self

            self._update_params(sstats, fic)

            n_shrinked_states = self._shrinkage_operation(dry=(not self._shrink))

            # TODO: this part should be integrated into Monitor
            if n_shrinked_states > 0:
                if not self._shrink:
                    print("shrinked: not an optimal model")
                    return self

        warnings.warn("end fitting though not yet converged", RuntimeWarning)
        return self
