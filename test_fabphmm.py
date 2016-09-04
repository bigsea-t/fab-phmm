import unittest
import numpy as np
from fab_phmm import FABPHMM
from utils import *


class TestFABPHMM(unittest.TestCase):

    def random_model(self, n_match_states=2, n_ins_states=2):
        initprob = transprob = emitprob = None

        return FABPHMM(n_match_states=n_match_states,
                             n_ins_states=n_ins_states)

    def a_model(self):
        n_match_states = 1
        n_ins_states = 1
        n_hstates = n_match_states + 2 * n_ins_states

        n_simbols = 4

        self.emptyprob = 1  # pseudou probability for unused column of emitprob frame

        self.matchprob = .16
        self.unmatchprob = .03
        self.insertprob = .25

        self.initprob_match = 0.6
        self.initprob_ins = 0.2

        initprob = np.array([self.initprob_match, self.initprob_ins, self.initprob_ins])

        transprob = np.array([[.9, .05, .05],
                              [.6, .4, 0],
                              [.6, 0, .4]])

        match_prob = np.ones((n_simbols, n_simbols)) * self.unmatchprob
        for i in range(n_simbols):
            match_prob[i, i] = self.matchprob

        emitprob = np.ones((n_hstates, n_simbols, n_simbols)) * self.insertprob

        emitprob[0, :, :] = match_prob

        return FABPHMM(n_match_states=n_match_states,
                       n_ins_states=n_ins_states,
                       initprob=initprob,
                       transprob=transprob,
                       emitprob=emitprob,
                       n_simbols=n_simbols)

    def sample_from_a_model(self, n_samples=1000, len_seq=30):
        model = self.a_model()
        xseqs = []
        yseqs = []

        for _ in range(n_samples):
            xseq, yseq, hseq = model.sample(len_seq)
            xseqs.append(omit_gap(xseq))
            yseqs.append(omit_gap(yseq))

        return xseqs, yseqs


    def test_fit(self):
        max_iter = 30
        xseqs, yseqs = self.sample_from_a_model(n_samples=1000, len_seq=10)
        model = self.random_model(n_match_states=1, n_ins_states=5)
        model.fit(xseqs, yseqs, max_iter=max_iter)


if __name__ == '__main__':
    unittest.main()
