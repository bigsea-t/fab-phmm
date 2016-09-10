from fab_phmm.fab_phmm import FABPHMM
import inspect
from experiments.utils import med_model, sample_from_model


def exec():
    """
    PHMM small
    sample from small mode and fit on PHMM
    """
    print(__name__)
    print(inspect.getsource(exec))

    smodel = med_model()
    fmodel = FABPHMM(n_match_states=1,
                     n_xins_states=10,
                     n_yins_states=10,
                     shrink_threshold=1e-2,
                     stop_threshold=1e-2)

    xseqs, yseqs = sample_from_model(smodel, n_samples=200, len_seq=30)
    fmodel.fit(xseqs, yseqs, max_iter=10000, verbose=True)


if __name__ == '__main__':
    exec()
