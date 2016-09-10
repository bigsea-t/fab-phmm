from fab_phmm.phmm import PHMM
from fab_phmm.utils import *
import warnings

class FABPHMM(PHMM):
    
    def __init__(self, n_match_states=1, n_xins_states=2, n_yins_states=2,
                 n_simbols=4, initprob=None, transprob=None, emitprob=None,
                 symmetric_emission=False, shrink_threshold=1e-2,
                 stop_threshold=1e-2):
        
        super(FABPHMM, self).__init__(n_match_states=n_match_states,
                                      n_xins_states=n_xins_states,
                                      n_yins_states=n_yins_states,
                                      n_simbols=n_simbols,
                                      initprob=initprob,
                                      transprob=transprob,
                                      emitprob=emitprob,
                                      stop_threshold=stop_threshold)
        # implement symmetric one for the first program

        self._symmetric_emission = symmetric_emission
        self._shrink_threshold = shrink_threshold
        self._qsum = None

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
        # dim_trans = np.sum(self._transprob != 0, axis=1) - 1
        # dim_init = np.sum(self._initprob != 0) - 1
        dim_trans = np.ones(n_hstates) * (n_hstates - 1)
        dim_init = n_hstates - 1

        for k in range(n_hstates):
            hstate_prop = self._hstate_properties[k]
            dim_emit[k] = self._dim_match_emit if hstate_prop == 0 else self._dim_ins_emit

        return dim_init, dim_trans, dim_emit


    def _gen_pseudo_prob(self, log_emitprob_frame, dims_trans, dims_emit):
        # TODO: two patterns of qsum are needed (for trans and for emit)
        term_trans = - dims_trans / (2 * self._qsum + EPS)
        term_emit = - dims_emit / (2 * self._qsum + EPS)

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

    def _shrinkage_operation(self):
        below_threshold = (self._qsum / np.sum(self._qsum)) < (self._shrink_threshold)
        preserved_hstates, = np.nonzero(~below_threshold)
        deleted_hstates, = np.nonzero(below_threshold)

        n_deleted_hstateprops = np.zeros(3, dtype=np.int)
        for k in deleted_hstates:
            n_deleted_hstateprops[self._hstate_properties[k]] += 1

        if np.sum(n_deleted_hstateprops) == 0:
            return 0

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
        self._qsum = self._qsum[preserved_hstates]

        for k in deleted_hstates:
            print("hstate {} is deleted".format(k))

        return np.sum(n_deleted_hstateprops)

    def _init_sufficient_statistics(self):
        sstats = super(FABPHMM, self)._init_sufficient_statistics()
        sstats["qsum"] = np.zeros(self._n_hstates)
        sstats["norm"] = 0
        return sstats

    def _accumulate_sufficient_statistics(self, sstats, gamma, xi, xseq, yseq, norm):
        super(FABPHMM, self)._accumulate_sufficient_statistics(sstats, gamma, xi, xseq, yseq)
        sstats["qsum"] += np.sum(gamma, axis=(0, 1))
        sstats["norm"] += np.sum(norm)

    def _gen_random_qsum(self, xseqs, yseqs):
        n_seqs = len(xseqs)
        qsum = np.zeros(self._n_hstates)
        for i in range(n_seqs):
            xlen, ylen = xseqs[i].shape[0], yseqs[i].shape[0]
            frame = np.random.rand(xlen + 1, ylen + 1, self._n_hstates)
            frame /= frame.sum(axis=2)[:, :, np.newaxis]
            qsum += np.sum(frame, axis=(0,1))
        return qsum

    def _update_params(self, sstats):
        super(FABPHMM, self)._update_params(sstats)
        self._qsum = sstats["qsum"]

    def _calculate_fic(self, sstats, free_energy, n_seq):
        dim_init, dims_trans, dims_emit = self._gen_dims()
        fic = free_energy + sstats["norm"] \
              + dim_init / 2 * np.log(n_seq) \
              + np.sum(dims_trans / 2 * np.log(sstats["qsum"]) -1) \
              + np.sum(dims_emit / 2 * np.log(sstats["qsum"]) - 1)

        return fic

    def fit(self, xseqs, yseqs, max_iter=1000, verbose=False):
        if not self._params_valid:
            self._params_random_init()
            self._params_valid = True

        assert(len(xseqs) == len(yseqs))

        self._qsum = self._gen_random_qsum(xseqs, yseqs)

        fic_prev = - np.inf

        for i in range(1, max_iter + 1):
            print("{}-th iteration...".format(i))
            log_transprob = log_(self._transprob)
            log_initprob = log_(self._initprob)

            sstats = self._init_sufficient_statistics()

            dim_init, dims_trans, dims_emit = self._gen_dims()

            fe_all = 0

            for j in range(len(xseqs)):
                log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])

                norm, pseudo_prob = self._gen_pseudo_prob(log_emitprob_frame, dims_trans, dims_emit)
                fab_log_emitprob_frame = log_emitprob_frame + log_(pseudo_prob)

                free_energy, fwd_lattice = self._forward(fab_log_emitprob_frame, log_transprob, log_initprob)

                fe_all += free_energy

                bwd_lattice = self._backward(fab_log_emitprob_frame, log_transprob)

                gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, free_energy,
                                                             log_emitprob_frame, log_transprob)

                self._accumulate_sufficient_statistics(sstats, gamma, xi, xseqs[j], yseqs[j], norm)

            fic = self._calculate_fic(sstats, fe_all, len(xseqs))

            if verbose:
                self._print_states(ll=fic, i_iter=i)

            if (fic - fic_prev) / len(xseqs) < self._stop_threshold:
                if fic - fic_prev < 0:
                    warnings.warn("fic decreased", RuntimeWarning)
                else:
                    print("end iteration with fic: {}".format(fic))
                    return fic

            fic_prev = fic

            self._update_params(sstats)

            n_shrinked_states = self._shrinkage_operation()

            if n_shrinked_states > 0:
                fic_prev = - np.inf
            print("n_hstates", self._n_hstates)
            print()

        return fic
