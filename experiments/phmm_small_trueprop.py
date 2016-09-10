from fab_phmm.phmm import PHMM
import inspect
from experiments.utils import small_model, sample_from_model


def main():
    """
    sample from small model and fit to PHMM
    start from true props
    """
    print(__name__)
    print(inspect.getsource(exec))

    smodel = small_model()
    fmodel = PHMM(n_match_states=1,
                  n_xins_states=1,
                  n_yins_states=1,
                  initprob=smodel._initprob,
                  transprob=smodel._transprob,
                  emitprob=smodel._emitprob)

    xseqs, yseqs = sample_from_model(smodel, n_samples=200, len_seq=30)
    fmodel.fit(xseqs, yseqs, max_iter=10000, verbose=True)

if __name__ == '__main__':
    main()






