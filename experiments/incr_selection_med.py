from fab_phmm.fab_phmm import incremental_model_selection
import inspect
from experiments.utils import med_model, sample_from_model, prepare_logd, get_dir
import pickle


def main(path_logd):
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

    models = incremental_model_selection(xseqs, yseqs, stop_threshold=1e-4, shrink_threshold=1e-2,
                                         max_iter=200, verbose=True, verbose_level=1,
                                         max_n_states=5)

    for i in range(min(3, len(models))):
        print("{} th model".format(i))
        models[i]._print_states()

    pickle.dump(models, open(path_logd + "models.pkl", "wb"))

    print("detailed model")
    models[0]._print_states(verbose_level=2)


if __name__ == '__main__':
    main(get_dir())
