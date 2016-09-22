from fab_phmm.fab_phmm import decremental_greedy_selection
import inspect
from experiments.utils import med_model, sample_from_model, prepare_logd, get_dir
import pickle

def main(path_logd=""):
    """
    incremental model selection
    model size: med
    """
    print(__name__)
    print(inspect.getsource(main))

    if path_logd != "":
        path_logd = prepare_logd(path_logd)

    smodel = med_model()
    xseqs, yseqs = sample_from_model(smodel, n_samples=200, len_seq=50)

    models = decremental_greedy_selection(xseqs, yseqs, stop_threshold=1e-4, shrink_threshold=1e-2,
                                         max_iter=200, verbose=True, verbose_level=1,
                                         max_n_states=5, sorted=False)

    monitor = models[-1].monitor

    pickle.dump(monitor, open(path_logd + "monitor.pkl", "wb"))
    pickle.dump(models, open(path_logd + "models.pkl", "wb"))

    models.sort(key=lambda m: - m._last_score)

    print("detailed model")
    models[0]._print_states(verbose_level=2)

if __name__ == '__main__':
    main(get_dir())
