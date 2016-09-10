from fab_phmm.phmm import PHMM
import numpy as np
from fab_phmm.utils import omit_gap


def _small_model():
    n_match_states = 1
    n_ins_states = 1
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    matchprob = .16
    unmatchprob = .03
    insertprob = .25

    initprob_match = 0.6
    initprob_ins = 0.2

    initprob = np.array([initprob_match, initprob_ins, initprob_ins])

    transprob = np.array([[.8, .1, .1],
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


def small_model():
    n_match_states = 1
    n_ins_states = 1
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    matchprob = .22
    unmatchprob = .01
    insertprob = .25

    initprob_match = 0.6
    initprob_ins = 0.2

    initprob = np.array([initprob_match, initprob_ins, initprob_ins])

    transprob = np.array([[.8, .1, .1],
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


def emit_only_model():
    n_match_states = 2
    n_ins_states = 2
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    initprob = np.array([.4, .6, 0, 0, 0, 0])

    transprob = np.array([[.8, .2, 0, 0, 0, 0],
                          [.2, .8, 0, 0, 0, 0],
                          [.9, .1, 0,  0,  0,  0],
                          [.9, .1,  0, 0,  0,  0],
                          [.9, .1,  0,  0, 0,  0],
                          [.9, .1,  0,  0,  0, 0]])

    matchprob = .16
    unmatchprob = .03
    insertprob = .25

    match_prob = np.ones((n_simbols, n_simbols)) * unmatchprob
    for i in range(n_simbols):
        match_prob[i, i] = matchprob

    emitprob = np.ones((n_hstates, n_simbols, n_simbols)) * insertprob

    emitprob[0, :, :] = match_prob

    matchprob = .10
    unmatchprob = .05

    match_prob = np.ones((n_simbols, n_simbols)) * unmatchprob
    for i in range(n_simbols):
        match_prob[i, i] = matchprob

    emitprob[1, :, :] = match_prob

    return PHMM(n_match_states=n_match_states,
                n_xins_states=n_ins_states,
                n_yins_states=n_ins_states,
                initprob=initprob,
                transprob=transprob,
                emitprob=emitprob,
                n_simbols=n_simbols)


def med_model():
    n_match_states = 1
    n_ins_states = 2
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    initprob = np.array([.8, .05, .05, .05, .05])

    transprob = np.array([[.8, .05, .05, .05, .05],
                          [.8, .2, 0,  0,  0],
                          [.2, 0, .8,  0,  0],
                          [.8, 0,  0,  .2, 0],
                          [.2, 0,  0,  0,  .8]])

    matchprob = .22
    unmatchprob = .01
    insertprob = .25

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


def med2_model():
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

    matchprob = .22
    unmatchprob = .01
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
