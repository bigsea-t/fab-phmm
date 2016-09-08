from fab_phmm.phmm import PHMM
from fab_phmm.fab_phmm import FABPHMM
import numpy as np
from fab_phmm.utils import omit_gap


def small_model():
    n_match_states = 1
    n_ins_states = 1
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    emptyprob = 1  # pseudou probability for unused column of emitprob frame

    matchprob = .16
    unmatchprob = .03
    insertprob = .25

    initprob_match = 0.6
    initprob_ins = 0.2

    initprob = np.array([initprob_match, initprob_ins, initprob_ins])

    transprob = np.array([[.9, .05, .05],
                          [.6, .4, 0],
                          [.6, 0, .4]])

    match_prob = np.ones((n_simbols, n_simbols)) * unmatchprob
    for i in range(n_simbols):
        match_prob[i, i] = matchprob

    emitprob = np.ones((n_hstates, n_simbols, n_simbols)) * insertprob

    emitprob[0, :, :] = match_prob

    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_ins_states,
                   n_yins_states=n_ins_states,
                   initprob=initprob,
                   transprob=transprob,
                   emitprob=emitprob,
                   n_simbols=n_simbols)

def med_model():
    n_match_states = 2
    n_ins_states = 2
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    initprob = np.array([.3, .3, .1, .1, .1, .1])

    transprob = np.array([[.3, .3, .1, .1, .1, .1],
                          [.4, .4, .05, .05, .05, .05],
                          [.4, .4, .2,  0,  0,  0],
                          [.2, .2,  0, .6,  0,  0],
                          [.4, .4,  0,  0, .2,  0],
                          [.2, .2,  0,  0,  0, .6]])

    matchprob = .16
    unmatchprob = .03
    insertprob = .25

    match_prob = np.ones((n_simbols, n_simbols)) * unmatchprob
    for i in range(n_simbols):
        match_prob[i, i] = matchprob

    emitprob = np.ones((n_hstates, n_simbols, n_simbols)) * insertprob

    emitprob[[0, 1], :, :] = match_prob


    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_ins_states,
                   n_yins_states=n_ins_states,
                   initprob=initprob,
                   transprob=transprob,
                   emitprob=emitprob,
                   n_simbols=n_simbols)



def sample_from_model(model, n_samples=1000, len_seq=30):
    xseqs = []
    yseqs = []

    for _ in range(n_samples):
        xseq, yseq, hseq = model.sample(len_seq)
        xseqs.append(omit_gap(xseq))
        yseqs.append(omit_gap(yseq))
        assert (xseqs[-1].shape[0] > 0)
        assert (yseqs[-1].shape[0] > 0)

    return xseqs, yseqs


max_iter = 10

smodel = small_model()
xseqs, yseqs = sample_from_model(smodel, n_samples=100, len_seq=20)

# mmodel = med_model()
# xseqs, yseqs = sample_from_model(mmodel, n_samples=400, len_seq=20)



# model = PHMM(n_match_states=1, n_xins_states=5, n_yins_states=1)
# model.fit(xseqs, yseqs, max_iter=max_iter)

fab = FABPHMM(n_match_states=1, n_xins_states=3, n_yins_states=3)
fab.fit(xseqs, yseqs, max_iter=max_iter)