import unittest
import numpy as np
import phmm
from utils import *


class TestPHMM(unittest.TestCase):

    def setUp(self):
        self.emptyprob = 1  # pseudou probability for unused column of emitprob frame

        self.matchprob = .85
        self.unmatchprob = .05
        self.insertprob = .25

        self.initprob_match = 0.9
        self.initprob_ins = 0.05

        n_match_states = 1
        n_ins_states = 1
        n_hstates = n_match_states + 2 * n_ins_states

        n_simbols = 4

        initprob = np.array([self.initprob_match, self.initprob_ins, self.initprob_ins])
        transprob = np.array([[.9, .05, .05],
                              [.6, .4, 0],
                              [.6, 0, .4]])

        match_prob = np.ones((n_simbols, n_simbols)) * self.unmatchprob
        for i in range(n_simbols):
            match_prob[i, i] = self.matchprob

        emitprob = np.ones((n_hstates, n_simbols, n_simbols)) * self.insertprob

        emitprob[0, :, :] = match_prob

        self.model = phmm.PHMM(n_match_states=n_match_states,
                          n_ins_states=1,
                          initprob=initprob,
                          transprob=transprob,
                          emitprob=emitprob,
                          n_simbols=n_simbols)

    def test_decoder(self):
        def decode(xseq, yseq):

            ll, map_hstates = self.model.decode(xseq, yseq)

            return map_hstates

        xseq = np.array([0, 0])
        yseq = np.array([0, 1, 0])
        ans = np.array([0, 2, 0])

        np.testing.assert_array_equal(ans, decode(xseq, yseq))

        xseq = np.array([1, 0, 0])
        yseq = np.array([0, 0])
        ans = np.array([1, 0, 0])

        np.testing.assert_array_equal(ans, decode(xseq, yseq))

        xseq = np.array([0, 0])
        yseq = np.array([0, 0, 1, 2, 3])
        ans = np.array([0, 0, 2, 2, 2])

        np.testing.assert_array_equal(ans, decode(xseq, yseq))

        xseq = np.array([0, 1, 2, 0, 1, 3, 0, 0, 0])
        yseq = np.array([0, 1, 2, 0, 3, 0, 0, 0])
        ans = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(ans, decode(xseq, yseq))

    def test_gen_log_emitprob_frame(self):

        def _gen_ans(xseq, yseq):
            len_x = xseq.shape[0]
            len_y = yseq.shape[0]
            n_hstates = self.model._n_hstates

            emitprob = np.ones((len_x + 1, len_y +1, n_hstates)) * self.emptyprob

            for i in range(len_x):
                for j in range(len_y):
                    if xseq[i] == yseq[j]:
                        emitprob[i+1, j+1, 0] = self.matchprob
                    else:
                        emitprob[i+1, j+1, 0] = self.unmatchprob

                    emitprob[i+1, j+1, [1,2]] = self.insertprob

            for i in range(len_x):
                emitprob[i + 1, 0, 1] = self.insertprob

            for j in range(len_y):
                emitprob[0, j + 1, 2] = self.insertprob

            return np.log(emitprob)

        def _test_if_correct(xseq, yseq):
            ret = self.model._gen_log_emitprob_frame(xseq, yseq)
            ans = _gen_ans(xseq, yseq)

            np.testing.assert_almost_equal(ans, ret)

        xseq = np.array([0,1])
        yseq = np.array([0,2,1])
        _test_if_correct(xseq, yseq)

        xseq = np.array([0,1,2,3,0,1,2,3])
        yseq = np.array([0,0,0,0,2])
        _test_if_correct(xseq, yseq)

        xseq = np.array([0])
        yseq = np.array([1])
        _test_if_correct(xseq, yseq)


    def test_forward(self):
        xseq = np.array([0])
        yseq = np.array([1])

        log_emitprob_frame = self.model._gen_log_emitprob_frame(xseq, yseq)
        log_transprob = log_(self.model._transprob)
        log_initprob = log_(self.model._initprob)

        ll, fwd_frame = self.model._forward(log_emitprob_frame, log_transprob, log_initprob)

        match11 = self.initprob_match * self.unmatchprob
        match_frame = log_(np.array([[0, 0], [0, match11]]))

        xins10 = self.initprob_ins * self.insertprob
        xins_frame = log_(np.array([[0, 0], [xins10, 0]]))

        yins01 = self.initprob_ins * self.insertprob
        yins_frame = log_(np.array([[0, yins01], [0, 0]]))

        ans = np.zeros_like(fwd_frame)

        ans[:, :, 0] = match_frame
        ans[:, :, 1] = xins_frame
        ans[:, :, 2] = yins_frame

        np.testing.assert_array_almost_equal(fwd_frame, ans)

    def test_backward(self):
        pass

    def test_fit(self):
        N = 100
        # xseqs = np.array([[0,1,2,3,0,1,2,3] for _ in range(N)])
        # yseqs = np.array([[0,1,2,0,1,2,3] for _ in range(N)])

        xseqs = np.array([[0,1] for _ in range(N)])
        yseqs = np.array([[0,2,1] for _ in range(N)])
        self.model.fit(xseqs, yseqs, max_iter=1)
        print("initprob", self.model._initprob)
        print("transprob", self.model._transprob)

        self.setUp()


if __name__ == '__main__':
    unittest.main()
