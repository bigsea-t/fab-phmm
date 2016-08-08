import unittest
import numpy as np
from phmm import PHMM


class TestPHMM(unittest.TestCase):

    def setUp(self):
        initprob = np.array([.9, .05, .05])
        transprob = np.array([[.9, .05, .05],
                             [.6, .4, 0],
                             [.6, 0, .4]])
        emitprob = np.array([[[.85, .05, .05, .05],
                              [.05, .85, .05, .05],
                              [.05, .05, .85, .05],
                              [.05, .05, .05, .05]],
                             [[.25, .25, .25, .25],
                              [.25, .25, .25, .25],
                              [.25, .25, .25, .25],
                              [.25, .25, .25, .25]],
                             [[.25, .25, .25, .25],
                              [.25, .25, .25, .25],
                              [.25, .25, .25, .25],
                              [.25, .25, .25, .25]]])

        self.model = PHMM(n_match_states=1,
                     n_ins_states=1,
                     initprob=initprob,
                     transprob=transprob,
                     emitprob=emitprob)

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

        xseq = np.array([0, 1, 2, 0, 1, 3, 0, 0, 0])
        yseq = np.array([0, 1, 2, 0, 3, 0, 0, 0])
        ans = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(ans, decode(xseq, yseq))


if __name__ == '__main__':
    unittest.main()
