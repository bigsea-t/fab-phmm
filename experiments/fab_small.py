from fab_phmm.fab_phmm import FABPHMM
import inspect
from experiments.utils import small_model, sample_from_model


def main():
    """
    PHMM small
    sample from small mode and fit on PHMM
    """
    print(__name__)
    print(inspect.getsource(main))

    smodel = small_model()
    fmodel = FABPHMM(n_match_states=1,
                     n_xins_states=10,
                     n_yins_states=10,
                     shrink_threshold=1e-2,
                     stop_threshold=1e-6)

    xseqs, yseqs = sample_from_model(smodel, n_samples=200, len_seq=30)
    fmodel.fit(xseqs, yseqs, max_iter=10000, verbose=True)
    print("end with n_hstates {}".format(fmodel._n_hstates))

if __name__ == '__main__':
    main()
