import pickle
import fnmatch
import os
import sys
import argparse

answers = {
    "small": (1, 1, 1),
    "med": (1, 2, 2)
}


def main(expname, dirname, size):
    results = []

    ans = answers[size]
    for folder in os.listdir(dirname):
        if fnmatch.fnmatch(folder, "*" + expname):
            datafile = os.path.join(dirname, folder, "models.pkl")
            if not os.path.exists(datafile):
                print("datafile: {} doesn't exist".format(datafile), file=sys.stderr)
                continue
            results.append(pickle.load(open(datafile, "rb")))

    n_corr = 0
    n_exp = len(results)

    for models in results:
        best_model = max(models, key=lambda x: x._last_score)
        n_match = best_model._n_match_states
        n_xins = best_model._n_xins_states
        n_yins = best_model._n_yins_states
        if(n_match, n_xins, n_yins) == ans:
            n_corr += 1
    print("{} correct models out of {} modles".format(n_corr, n_exp))
    print("accuracy: {}".format(n_corr / n_exp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str,
                        help="name of data directory")

    parser.add_argument("-n", "--name", type=str,
                        help="name of experiment")

    parser.add_argument("-s", "--size", type=str,
                        help="med/small")

    args = parser.parse_args()
    main(args.name, args.directory, args.size)