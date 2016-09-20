from fab_phmm.fab_phmm import FABPHMM
from fab_phmm.phmm import PHMM
import inspect
from experiments.utils import med_model, sample_from_model, med2_model, med3_model


def main():
    """
    PHMM med
    sample from med mode and fit on FABPHMM without shrinkage
    """
    print(__name__)
    print(inspect.getsource(main))

    smodel = med3_model()
    xseqs, yseqs = sample_from_model(smodel, n_samples=200, len_seq=50)

    fab_small = FABPHMM(n_match_states=1,
                     n_xins_states=1,
                     n_yins_states=1,
                     shrink_threshold=1e-2,
                     stop_threshold=1e-4,
                     shrink=False)

    fab_med = FABPHMM(n_match_states=1,
                     n_xins_states=2,
                     n_yins_states=2,
                     shrink_threshold=1e-2,
                     stop_threshold=1e-4,
                     shrink=False)

    phmm_small = PHMM(n_match_states=1,
                      n_xins_states=1,
                      n_yins_states=1,
                      stop_threshold=1e-4)

    phmm_med = PHMM(n_match_states=1,
                      n_xins_states=2,
                      n_yins_states=2,
                      stop_threshold=1e-4)



    fic_small = fab_small.fit(xseqs, yseqs, max_iter=500, verbose=True, verbose_level=1)
    ll_small = fab_small.score(xseqs, yseqs, type="ll")

    fic_med =  fab_med.fit(xseqs, yseqs, max_iter=500, verbose=True, verbose_level=1)
    ll_med = fab_med.score(xseqs, yseqs, type="ll")

    phmm_small.fit(xseqs, yseqs, max_iter=500, verbose=True, verbose_level=1)
    ll_phmms = phmm_small.score(xseqs, yseqs)
    phmm_med.fit(xseqs, yseqs, max_iter=500, verbose=True, verbose_level=1)
    ll_phmmm = phmm_med.score(xseqs, yseqs)

    ll_org = smodel.score(xseqs, yseqs)
    smodel.fit(xseqs, yseqs)
    ll_org_fit = smodel.score(xseqs, yseqs)

    print("dump fab_small")
    fab_small._print_states(verbose_level=2)
    print("dump fab_med")
    fab_med._print_states(verbose_level=2)
    print("dump phmm_small")
    phmm_small._print_states(verbose_level=2)
    print("dump phmm_med")
    phmm_med._print_states(verbose_level=2)

    print("fic_small", fic_small)
    print("ll_small", ll_small)
    print("fic_med", fic_med)
    print("ll_med", ll_med)
    print("ll_phmms", ll_phmms)
    print("ll_phmmm", ll_phmmm)
    print("ll_org", ll_org)
    print("ll_org", ll_org_fit)

if __name__ == '__main__':
    main()
