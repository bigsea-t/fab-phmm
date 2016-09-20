from fab_phmm.fab_phmm import incremental_model_selection
import inspect
from experiments.utils import med_model, sample_from_model


def main():
    """
    incremental model selection
    model size: med
    """
    print(__name__)
    print(inspect.getsource(main))

    smodel = med_model()
    xseqs, yseqs = sample_from_model(smodel, n_samples=200, len_seq=50)

    models = incremental_model_selection(xseqs, yseqs, stop_threshold=1e-4, shrink_threshold=1e-2,
                                         max_iter=200, verbose=True, verbose_level=0,
                                         max_n_states=10)

    for i in range(min(3, len(models))):
        print("{} th model".format(i))
        models[i]._print_states()


if __name__ == '__main__':
    main()
