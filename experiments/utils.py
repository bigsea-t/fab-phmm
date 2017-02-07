from fab_phmm.phmm import PHMM
from fab_phmm.fab_phmm import  FABPHMM
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

    transprob = np.array([[.8, .07, .03, .07, .03],
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

    transprob = np.array([[.78, .1, .04, .02, .04, .02],
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

def large_model():
    n_match_states = 3
    n_ins_states = 2
    n_hstates = n_match_states + 2 * n_ins_states

    n_simbols = 4

    initprob = np.array([.5, .1, .2, .08, .02, .08, .02])

    transprob = np.array([[.58, .1, .1, .04, .02, .04, .12],
                          [.05, .9, .01, .01, .01, .01, .01],
                          [.1, .3, .1, .24, .1, .14, .02],
                          [.5, .1, .1, .1,  0,  0,  0],
                          [.1, .3, .2, 0, .4,  0,  0],
                          [.1, .3, .5, 0,  0, .1,  0],
                          [.2, .2, .2, 0,  0,  0, .4]])

    match1 = np.array([[.1, 0, 0, 0],
                       [0, .1, 0, 0],
                       [0, 0, .1, 0],
                       [0, 0, 0, .8]])

    match2 = np.array([[.07, .01, .01, .01],
                       [.01, .07, .01, .01],
                       [.01, .01, .57, .01],
                       [.01, .01, .01, .17]])

    match3 = np.array([[.37, .01, .01, .01],
                       [.01, .37, .01, .01],
                       [.01, .01, .07, .01],
                       [.01, .01, .01, .07]])

    ins_a_x = np.ones((n_simbols, n_simbols)) * np.array([.5, .4, .05, .05])[:, np.newaxis]
    ins_a_y = ins_a_x.T
    ins_b_x = np.ones((n_simbols, n_simbols)) * np.array([.05, .05, .4, .5])[:, np.newaxis]
    ins_b_y = ins_b_x.T

    emitprob = np.zeros((n_hstates, n_simbols, n_simbols))

    emitprob[0, :, :] = match1
    emitprob[1, :, :] = match2
    emitprob[2, :, :] = match3
    emitprob[3, :, :] = ins_a_x
    emitprob[4, :, :] = ins_b_x
    emitprob[5, :, :] = ins_a_y
    emitprob[6, :, :] = ins_b_y

    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_ins_states,
                   n_yins_states=n_ins_states,
                   initprob=initprob,
                   transprob=transprob,
                   emitprob=emitprob,
                   n_simbols=n_simbols)

def huge_model():
    n_match_states = 5
    n_xins_states = 2
    n_yins_states = 1

    n_simbols = 4

    initprob = np.array([1.52250209e-01, 1.73901308e-01, 2.12584427e-01,
                         2.23262778e-01, 1.51633394e-01,   4.38939224e-02,
         4.24739609e-02,   2.13068007e-26])

    transprob = np.array([[8.10682058e-02, 5.20547306e-01, 2.77782021e-01,
            4.79388383e-05, 5.73136913e-02, 1.91572066e-02,
            2.28284185e-02, 2.12552123e-02],
           [3.23034774e-01, 2.90914732e-01, 2.99734944e-01,
            1.31605054e-02, 2.14674649e-02, 2.55636094e-02,
            1.08771769e-02, 1.52467930e-02],
           [6.29750882e-01, 5.99684915e-08, 3.21309829e-01,
            1.65327737e-04, 1.89662317e-02, 4.41917044e-03,
            7.40436483e-03, 1.79841351e-02],
           [2.52164440e-03, 4.66846900e-03, 1.00893950e-02,
            8.65158118e-01, 7.53603488e-02, 2.98792557e-02,
            6.25705646e-04, 1.16970631e-02],
           [4.48524703e-02, 2.03525802e-02, 8.45472416e-02,
            8.37729739e-02, 7.35089193e-01, 1.56998528e-02,
            1.92334615e-03, 1.37623425e-02],
           [7.88854838e-06, 1.04723049e-01, 4.88676478e-02,
            8.48254908e-02, 1.02709792e-01, 6.58866132e-01,
            0.00000000e+00, 0.00000000e+00],
           [2.28500256e-03, 2.50986396e-02, 7.20435122e-02,
            6.95896615e-05, 1.40030880e-03, 0.00000000e+00,
            8.99102947e-01, 0.00000000e+00],
           [6.85894986e-02, 2.80908651e-02, 6.05330847e-02,
            3.21919966e-02, 3.27113854e-02, 0.00000000e+00,
            0.00000000e+00, 7.77883170e-01]])

    emitprob = np.array([[[4.40086542e-03, 7.14642286e-02, 2.28578318e-02, 1.63966800e-03],
        [1.70746562e-02,   3.88626536e-01,   4.55716187e-02,
           5.27847677e-10],
        [  7.20770705e-03,   3.01580599e-02,   3.09579378e-01,
           1.24607445e-02],
        [  3.80436237e-21,   2.44972642e-02,   6.08879067e-02,
           3.57353442e-03]],

       [[  5.06862412e-01,   1.09158647e-01,   4.48658641e-02,
           2.39580174e-02],
        [  1.36818034e-01,   5.35084894e-05,   7.20991883e-05,
           1.33164573e-03],
        [  4.43529434e-02,   7.38204315e-03,   4.65462409e-02,
           3.05850450e-04],
        [  4.52169843e-02,   4.02936706e-03,   2.84035704e-02,
           6.42772321e-04]],

       [[  3.13647232e-06,   3.01377254e-02,   1.23360820e-04,
           3.08615105e-02],
        [  3.00667529e-04,   1.76152900e-01,   7.29051762e-03,
           4.30562956e-02],
        [  7.34084625e-06,   3.24690923e-05,   1.48346629e-02,
           1.09601252e-01],
        [  1.69782347e-02,   5.34127994e-02,   9.55213896e-02,
           4.21685737e-01]],

       [[  9.07112819e-02,   3.83118861e-02,   1.77610784e-02,
           4.71015547e-03],
        [  5.94417277e-02,   4.15133866e-01,   3.30740455e-02,
           2.94775779e-02],
        [  1.68330679e-02,   2.29128233e-02,   1.60927935e-01,
           2.76283268e-02],
        [  3.18583945e-03,   1.49696712e-02,   3.11308911e-02,
           3.37898255e-02]],

       [[  1.65057479e-02,   1.60187447e-02,   1.54342918e-02,
           1.19093828e-02],
        [  2.37819064e-02,   5.81066977e-02,   2.63616219e-02,
           1.13652863e-02],
        [  2.87327021e-02,   3.53167411e-02,   5.00302181e-01,
           7.36873815e-02],
        [  1.15195462e-02,   9.07015696e-03,   6.61471172e-02,
           9.57404942e-02]],

       [[  1.72774419e-01,   1.72774419e-01,   1.72774419e-01,
           1.72774419e-01],
        [  3.68970649e-01,   3.68970649e-01,   3.68970649e-01,
           3.68970649e-01],
        [  3.52549866e-01,   3.52549866e-01,   3.52549866e-01,
           3.52549866e-01],
        [  1.05705066e-01,   1.05705066e-01,   1.05705066e-01,
           1.05705066e-01]],

       [[  2.43306155e-01,   2.43306155e-01,   2.43306155e-01,
           2.43306155e-01],
        [  2.31434163e-01,   2.31434163e-01,   2.31434163e-01,
           2.31434163e-01],
        [  2.62364505e-01,   2.62364505e-01,   2.62364505e-01,
           2.62364505e-01],
        [  2.62895176e-01,   2.62895176e-01,   2.62895176e-01,
           2.62895176e-01]],

       [[  1.96393214e-01,   2.91945036e-01,   3.04948763e-01,
           2.06712987e-01],
        [  1.96393214e-01,   2.91945036e-01,   3.04948763e-01,
           2.06712987e-01],
        [  1.96393214e-01,   2.91945036e-01,   3.04948763e-01,
           2.06712987e-01],
        [  1.96393214e-01,   2.91945036e-01,   3.04948763e-01,
           2.06712987e-01]]])

    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_xins_states,
                   n_yins_states=n_yins_states,
                   initprob=initprob,
                   transprob=transprob,
                   emitprob=emitprob,
                   n_simbols=n_simbols)

def m533_model():
    n_match_states = 5
    n_xins_states = 3
    n_yins_states = 3

    n_simbols = 4

    initprob = np.array([  1.74025574e-001,   2.02487037e-001,   9.02697286e-002,
         1.65972599e-001,   2.86802745e-001,   1.50094058e-002,
         2.51820392e-002,   4.02508702e-002,   1.53121845e-039,
         6.18009805e-135,   4.68295749e-071])


    transprob = np.array([[2.29870373e-01, 2.36430363e-01, 8.40144731e-02,
            1.73321989e-01, 1.96457664e-01, 5.96742117e-03,
            3.33349159e-02, 8.15858501e-03, 1.87202145e-02,
            8.59183666e-03, 5.13216415e-03],
           [5.01312337e-12, 2.28952298e-01, 4.07828833e-01,
            2.55801116e-01, 9.79197540e-02, 2.41478407e-03,
            6.52019556e-07, 9.02461283e-09, 1.55197808e-03,
            4.92408350e-03, 6.06492092e-04],
           [3.07117325e-01, 2.35764445e-01, 4.43833037e-03,
            1.23780006e-01, 1.87954692e-01, 7.78406044e-04,
            6.88754357e-02, 1.14351806e-02, 4.52511180e-02,
            1.46050616e-02, 3.21832829e-15],
           [7.14116703e-01, 1.53998640e-01, 1.60955988e-04,
            6.97953684e-08, 1.25508826e-01, 1.80708586e-07,
            3.71067294e-04, 9.28372641e-04, 4.88889236e-03,
            2.62923810e-05, 5.02089294e-10],
           [5.17057140e-05, 1.66285304e-01, 3.13551009e-07,
            1.95758241e-01, 5.84834189e-01, 9.88169375e-07,
            1.99206013e-02, 1.33119134e-02, 6.18591453e-05,
            1.92128225e-02, 5.62062197e-04],
           [1.79693070e-02, 2.00950692e-02, 1.40127671e-20,
            9.43293510e-03, 1.19826799e-06, 9.52501490e-01,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00],
           [2.98527289e-02, 1.50393055e-01, 5.06471644e-03,
            1.58615219e-01, 1.10361795e-01, 0.00000000e+00,
            5.45712485e-01, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00],
           [7.59442172e-03, 2.18456041e-02, 5.46901923e-03,
            1.76449076e-02, 3.96863886e-02, 0.00000000e+00,
            0.00000000e+00, 9.07759659e-01, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00],
           [4.67442572e-02, 8.98702895e-02, 1.65756948e-06,
            8.31517001e-02, 2.08608060e-11, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 7.80232096e-01,
            0.00000000e+00, 0.00000000e+00],
           [7.15712398e-03, 6.52943502e-02, 1.88623622e-07,
            3.45519618e-02, 1.51034432e-01, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            7.41961943e-01, 0.00000000e+00],
           [1.31535218e-02, 4.08240365e-03, 1.21362185e-08,
            8.53321007e-04, 1.24688602e-02, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 9.69441881e-01]])

    emitprob = np.array([[[1.12934790e-06, 5.11357725e-08, 1.35402425e-11,
             5.09960520e-02],
            [7.84216424e-03, 4.05916075e-04, 8.32608389e-07,
             5.58257158e-02],
            [2.09182444e-04, 9.68168139e-05, 3.35980195e-03,
             1.58042086e-01],
            [2.56129734e-02, 4.03960988e-02, 1.05814116e-01,
             5.51397063e-01]],

           [[3.39180663e-01, 1.03591025e-01, 5.55270296e-02,
             1.03365937e-02],
            [3.39376049e-02, 3.17637465e-05, 2.50436596e-03,
             7.77639792e-24],
            [4.04508063e-02, 1.00965494e-02, 3.37809949e-01,
             9.65488908e-05],
            [2.21679659e-02, 2.61446333e-03, 3.76767280e-02,
             3.97794312e-03]],

           [[1.92812813e-02, 4.19521933e-02, 2.76722186e-02,
             1.15564816e-04],
            [7.23901117e-02, 1.13845299e-03, 3.93167074e-02,
             3.49112068e-25],
            [3.57959261e-02, 5.94701879e-02, 6.47307074e-01,
             5.55602258e-02],
            [3.52450725e-11, 3.43522138e-18, 5.58084883e-08,
             1.40277494e-14]],

           [[1.34720910e-19, 6.57493920e-02, 1.23460351e-02,
             6.04621213e-04],
            [3.41879316e-13, 4.19824798e-01, 4.06314768e-02,
             1.06237294e-02],
            [3.34247036e-21, 3.87406507e-02, 2.82440687e-01,
             1.06737312e-06],
            [2.82413472e-04, 2.97719200e-02, 8.21702019e-02,
             1.68130066e-02]],

           [[2.06528696e-01, 6.20537625e-02, 1.27373567e-02,
             2.24556009e-03],
            [8.65655521e-02, 4.22946647e-01, 2.16098830e-02,
             9.36881641e-03],
            [1.34106238e-02, 1.24677709e-02, 8.08148578e-02,
             4.77177240e-03],
            [1.21511972e-02, 1.28660242e-02, 3.22060210e-02,
             7.25545857e-03]],

           [[3.53924437e-01, 3.53924437e-01, 3.53924437e-01,
             3.53924437e-01],
            [1.36852092e-01, 1.36852092e-01, 1.36852092e-01,
             1.36852092e-01],
            [1.52331030e-01, 1.52331030e-01, 1.52331030e-01,
             1.52331030e-01],
            [3.56892441e-01, 3.56892441e-01, 3.56892441e-01,
             3.56892441e-01]],

           [[1.91320662e-01, 1.91320662e-01, 1.91320662e-01,
             1.91320662e-01],
            [2.23912507e-01, 2.23912507e-01, 2.23912507e-01,
             2.23912507e-01],
            [3.14486347e-01, 3.14486347e-01, 3.14486347e-01,
             3.14486347e-01],
            [2.70280484e-01, 2.70280484e-01, 2.70280484e-01,
             2.70280484e-01]],

           [[1.78773734e-01, 1.78773734e-01, 1.78773734e-01,
             1.78773734e-01],
            [3.46984152e-01, 3.46984152e-01, 3.46984152e-01,
             3.46984152e-01],
            [3.30984949e-01, 3.30984949e-01, 3.30984949e-01,
             3.30984949e-01],
            [1.43257165e-01, 1.43257165e-01, 1.43257165e-01,
             1.43257165e-01]],

           [[1.91986162e-01, 1.40829112e-01, 5.50749331e-01,
             1.16435395e-01],
            [1.91986162e-01, 1.40829112e-01, 5.50749331e-01,
             1.16435395e-01],
            [1.91986162e-01, 1.40829112e-01, 5.50749331e-01,
             1.16435395e-01],
            [1.91986162e-01, 1.40829112e-01, 5.50749331e-01,
             1.16435395e-01]],

           [[1.62355215e-01, 5.29650568e-01, 1.69767030e-01,
             1.38227187e-01],
            [1.62355215e-01, 5.29650568e-01, 1.69767030e-01,
             1.38227187e-01],
            [1.62355215e-01, 5.29650568e-01, 1.69767030e-01,
             1.38227187e-01],
            [1.62355215e-01, 5.29650568e-01, 1.69767030e-01,
             1.38227187e-01]],

           [[2.55109862e-01, 2.07452002e-01, 2.16013242e-01,
             3.21424894e-01],
            [2.55109862e-01, 2.07452002e-01, 2.16013242e-01,
             3.21424894e-01],
            [2.55109862e-01, 2.07452002e-01, 2.16013242e-01,
             3.21424894e-01],
            [2.55109862e-01, 2.07452002e-01, 2.16013242e-01,
             3.21424894e-01]]])

    return PHMM(n_match_states=n_match_states,
                   n_xins_states=n_xins_states,
                   n_yins_states=n_yins_states,
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
    elif size == "large":
        return large_model()
    elif size == "huge":
        return huge_model()
    elif size == "m533":
        return m533_model()
    raise ValueError("invalid size {}".format(size))