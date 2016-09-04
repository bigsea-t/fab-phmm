import numpy as np
from utils import *
from phmm import PHMM


class FABPHMM(PHMM):
    
    def __init__(self, n_match_states=1, n_ins_states=2,
                 n_simbols=4, initprob=None, transprob=None, emitprob=None,
                 symmetric_emission=False, shrink_threshold=1e-3):
        
        super(FABPHMM, self).__init__(n_match_states=n_match_states,
                                      n_ins_states=n_ins_states,
                                      n_simbols=n_simbols,
                                      initprob=initprob,
                                      transprob=transprob,
                                      emitprob=emitprob)
        # implement symmetric one for the first program

        self._symmetric_emission = symmetric_emission
        self._shrink_threshold = shrink_threshold

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


    def _gen_pseudo_prob(self, q_tilde, dims_trans, dims_emit):
        sum_qtilde = np.sum(q_tilde, axis=(0, 1))
        term_trans = - dims_trans / (2 * sum_qtilde + EPS)
        term_emit = - dims_emit / (2 * sum_qtilde + EPS)

        score_1 = np.exp(term_trans + term_emit)
        norm_1 = np.sum(score_1)
        score_2 = np.exp(term_emit)
        norm_2 = np.sum(score_2)

        xshape, yshape, n_hstates = q_tilde.shape

        norm = np.ones((xshape, yshape)) * norm_1
        norm[-1] = norm_2

        pseudo_prob = np.ones((xshape, yshape, n_hstates)) * (score_1 / norm_1)[np.newaxis, np.newaxis, :]
        pseudo_prob[-1, -1, :] = score_2 / norm_2

        return norm, pseudo_prob

    def _shrinkage_operation(self, sstats):
        below_threshold = (sstats["gamma"] / sstats["n_gamma"])  > self._shrink_threshold / self._n_hstates
        preserved_hstates, = np.nonzero(below_threshold == 0)
        deleted_hstates = np.nonzero(below_threshold != 0)

        deleted_hstateprops = [0] * 3
        for k in deleted_hstates:
            deleted_hstateprops[self._hstate_properties[k]] += 1

        # update hstate props
        # add xins and yins nstates

        # dont forget update qtilde
        # and hstate properties

        self._initprob = self._initprob[preserved_hstates]
        self._transprob = self._transprob[...]
        self._emitprob = self._emitprob[preserved_hstates, :, :]
        self.

        for k in deleted_hstates:
            print("hstate {} is shrinked".format(k))




    def _init_sufficient_statistics(self):
        sstats = super(FABPHMM, self)._init_sufficient_statistics()
        sstats["gamma"] = np.zeros(self._n_hstates)
        sstats["n_gamma"] = 0
        return sstats

    def _accumulate_sufficient_statistics(self, sstats, gamma, xi, xseq, yseq):
        super(FABPHMM, self)._accumulate_sufficient_statistics(sstats, gamma, xi, xseq, yseq)
        sstats["gamma"] += np.sum(gamma, axis=(0, 1))
        sstats["n_gamma"] += gamma.shape[0] * gamma.shape[0]

    def _gen_random_distribution(self, xseqs, yseqs):
        n_seqs = len(xseqs)
        out = []
        for i in range(n_seqs):
            xlen, ylen = xseqs[i].shape[0], yseqs[i].shape[0]
            frame = np.random.rand(xlen + 1, ylen + 1, self._n_hstates)
            frame /= frame.sum(axis=2)[:, :, np.newaxis]
            out.append(frame)
        return out

    def fit(self, xseqs, yseqs, max_iter=1000):
        if not self._params_valid:
            self._params_random_init()
            self._params_valid = True

        log_transprob = log_(self._transprob)
        log_initprob = log_(self._initprob)

        assert(len(xseqs) == len(yseqs))

        qs_prev = self._gen_random_distribution(xseqs, yseqs)

        for i in range(1, max_iter + 1):
            print("{}-th iteration...".format(i))

            sstats = self._init_sufficient_statistics()

            dim_init, dims_trans, dims_emit = self._gen_dims()

            for j in range(len(xseqs)):
                log_emitprob_frame = self._gen_log_emitprob_frame(xseqs[j], yseqs[j])
                q_tilde = qs_prev[j]

                norm, pseudo_prob = self._gen_pseudo_prob(q_tilde, dims_trans, dims_emit)
                fab_log_emitprob_frame = log_emitprob_frame + log_(pseudo_prob)

                free_energy, fwd_lattice = self._forward(fab_log_emitprob_frame, log_transprob, log_initprob)
                bwd_lattice = self._backward(fab_log_emitprob_frame, log_transprob)

                gamma, xi = self._compute_smoothed_marginals(fwd_lattice, bwd_lattice, free_energy,
                                                             log_emitprob_frame, log_transprob)

                qs_prev[j] = gamma

                self._accumulate_sufficient_statistics(sstats, gamma, xi, xseqs[j], yseqs[j])

            self._update_params(sstats)

            self._shrinkage_operation(sstats)

        return self
