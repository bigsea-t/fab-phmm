from fab_phmm.fab_phmm import FABPHMM
from experiments.utils import get_model_by_size, sample_from_model, prepare_logd
import pickle
import timeit
import argparse
import os

# TODO: Logger class


def time_train(model, xseqs, yseqs, n_iter):
    """
    return sec [float]
    """
    start = timeit.default_timer()
    model.fit(xseqs, yseqs,
              max_iter=n_iter, verbose=False)
    end = timeit.default_timer()

    return end - start


def format_bench_result(benchs, n_iter):
    s = """
==========================================
            FAB-PHMM benchmark
==========================================
        benchmark of EM iteration\n\n"""
    s += "n_iter: {}\n".format(n_iter)
    for name, sec in benchs.items():
        s += "{:<20}: {:<20} [sec]\n".format(name, sec)

    return s


def gen_model(n_states):
    return FABPHMM(n_match_states=n_states[0],
                   n_xins_states=n_states[1],
                   n_yins_states=n_states[2],
                   stop_threshold=-1, shrink_threshold=-1)


def main(n_iter, path_logd=""):
    if path_logd is None:
        path_logd = ""
    if path_logd != "":
        path_logd = prepare_logd(path_logd, redirect_std=False, timestamp=False)

    print("sampling")
    smodel = get_model_by_size("small")
    xseqs, yseqs = sample_from_model(smodel, n_samples=50, len_seq=50)

    # several models will be added
    name_nstates = {"small (1,1,1)": (1, 1, 1),
                    "med (2,2,2)": (2, 2, 2),
                    "large (5,5,5)": (5, 5, 5)}

    models = {}
    for name, n_states in name_nstates.items():
        m = gen_model(n_states)
        models[name] = m
    print("benchmark training")
    benchs = {}
    for name, model in models.items():
        sec = time_train(model, xseqs, yseqs, n_iter=n_iter)
        benchs[name] = sec

    print(format_bench_result(benchs, n_iter),
          file=open(os.path.join(path_logd, "benchs.txt"), 'w'))

    pickle.dump(benchs, open(os.path.join(path_logd, "benchs.pkl"), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str,
                        help="name of log directory")
    parser.add_argument("-n", "--n_iter", type=int,
                        help="num of iteration")
    args = parser.parse_args()

    assert(args.n_iter is not None)

    main(args.n_iter, path_logd=args.directory)