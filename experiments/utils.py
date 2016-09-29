from fab_phmm.phmm import PHMM
import numpy as np
from fab_phmm.utils import omit_gap
import os
import sys
import datetime
import argparse


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

    initprob = np.array([.4, .15, .15, .15, .15])

    matchprob = 0.235
    unmatchprob = 0.05

    transprob = np.array([[.8, .03, .07, .03, .07],
                          [.9, .1, 0,  0,  0],
                          [.1, 0, .9,  0,  0],
                          [.9, 0,  0,  .1, 0],
                          [.1, 0,  0,  0,  .9]])

    ins_a_x = np.ones((n_simbols, n_simbols)) * np.array([.5, .4, .05, .05])[:, np.newaxis]
    ins_a_y = ins_a_x.T
    ins_b_x = np.ones((n_simbols, n_simbols)) * np.array([.05, .05, .4, .5])[:, np.newaxis]
    ins_b_y = ins_b_x.T

    match_prob = np.ones((n_simbols, n_simbols)) * unmatchprob
    for i in range(n_simbols):
        match_prob[i, i] = matchprob

    emitprob = np.ones((n_hstates, n_simbols, n_simbols))

    emitprob[0, :, :] = match_prob
    emitprob[1, :, :] = ins_a_x
    emitprob[2, :, :] = ins_b_x
    emitprob[3, :, :] = ins_a_y
    emitprob[4, :, :] = ins_b_y

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

    initprob = np.array([.3, .3, .05, .15, .05, .15])

    transprob = np.array([[.78, .1, .02, .04, .02, .04],
                          [.1, .78, .04, .02, .04, .02],
                          [.45, .45, .1,  0,  0,  0],
                          [.3, .3,  0, .4,  0,  0],
                          [.45, .45,  0,  0, .1,  0],
                          [.3, .3,  0,  0,  0, .4]])


    match1 = np.array([[.4, 0, 0, 0],
                       [0, .4, 0, 0],
                       [0, 0, .1, 0],
                       [0, 0, 0, .1]])

    match2 = np.array([[.07, .01, .01, .01],
                       [.01, .07, .01, .01],
                       [.01, .01, .37, .01],
                       [.01, .01, .01, .37]])

    ins_a_x = np.ones((n_simbols, n_simbols)) * np.array([.4, .3, .2, .1])[:, np.newaxis]
    ins_a_y = ins_a_x.T
    ins_b_x = np.ones((n_simbols, n_simbols)) * np.array([.1, .2, .3, .4])[:, np.newaxis]
    ins_b_y = ins_b_x.T

    emitprob = np.zeros((n_hstates, n_simbols, n_simbols))

    emitprob[0, :, :] = match1
    emitprob[1, :, :] = match2
    emitprob[2, :, :] = ins_a_x
    emitprob[3, :, :] = ins_b_x
    emitprob[4, :, :] = ins_a_y
    emitprob[5, :, :] = ins_b_y

    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_ins_states,
                   n_yins_states=n_ins_states,
                   initprob=initprob,
                   transprob=transprob,
                   emitprob=emitprob,
                   n_simbols=n_simbols)


def med3_model():
    n_match_states = 2
    n_ins_states = 1
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    initprob = np.array([.4, .4, .1, .1,])

    transprob = np.array([[.88, .1, .01, .01],
                          [.1, .7, .15, .15]
                          [.01, .68, .31, 0],
                          [.01, 0.68, 0, .31]])

    match1 = np.array([[.4, 0, 0, 0],
                       [0, .4, 0, 0],
                       [0, 0, .1, 0],
                       [0, 0, 0, .1]])

    match2 = np.array([[.11, .01, .01, .01],
                       [.01, .11, .01, .01],
                       [.01, .01, .33, .01],
                       [.01, .01, .01, .33]])

    emitprob = np.zeros((n_hstates, n_simbols, n_simbols)) * .25

    emitprob[0, :, :] = match1
    emitprob[1, :, :] = match2

    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_ins_states,
                   n_yins_states=n_ins_states,
                   initprob=initprob,
                   transprob=transprob,
                   emitprob=emitprob,
                   n_simbols=n_simbols)


def sample_from_model(model, n_samples=1000, len_seq=30, max_iter=1000):
    xseqs = []
    yseqs = []

    for _ in range(max_iter):
        break_flg = True
        for _ in range(n_samples):
            xseq, yseq, hseq = model.sample(len_seq)
            xseqs.append(omit_gap(xseq))
            yseqs.append(omit_gap(yseq))
            if (xseqs[-1].shape[0] == 0) or (yseqs[-1].shape[0] == 0):
                break_flg = False
        if break_flg:
            break

    return xseqs, yseqs


def prepare_logd(path_logd, redirect_std=True, timestamp=True):

    if timestamp:
        path_logd = path_logd + "_" +datetime.datetime.now().isoformat()

    if os.path.exists(path_logd):
        raise ValueError("directory already exist")

    os.makedirs(path_logd)

    if redirect_std:
        sys.stdout = open(os.path.join(path_logd, 'stdout.txt'), 'w')
        sys.stderr = open(os.path.join(path_logd, 'stderr.txt'), 'w')

    return path_logd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str,
                        help="name of output directory")

    parser.add_argument("-s", "--size", type=str,
                        help="small/med")

    args = parser.parse_args()

    if args.size not in ["small", "med"]:
        if args.size is None:
            args.size = "med"
        else:
            ValueError("size should be small or med")

    if args.directory is None:
        args.directory = ""

    return args


def get_model_by_size(size="med"):
    if size == "small":
        return small_model()
    elif size == "med":
        return med_model()