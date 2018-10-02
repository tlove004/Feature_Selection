# interface.py implements the text UI
# format for text UI taken from project prompt
import util
from timeit import default_timer as timer
import argparse
import logging
import sys


def GetFile():
    """
    Get and parse data from the given file.
    :return: Parsed data.
    """
    print("Welcome to Tyson Loveless' Feature Selection Algorithm.")
    name = raw_input('Type in the name of the file to test: ')
    return util.parse(name)


def GetAlgorithm():
    """
    Allows the user to select an algorithm to run on a dataset
    :return: algorithm to be used.
    """
    print("Type the number of the algorithm you want to run.")
    print("\n   1)  Forward Selection")
    print("\n   2)  Backward Elimination")
    print("\n   3)  Tyson's Genetic Algorithm\n")
    search_type = input('                 ')
    return search_type


def get_algorithm(arg):
    if arg == 'ss':
        return 3
    elif arg == 'bs':
        return 2
    elif arg == 'fs':
        return 1


def main(args):
    """
    Main method of the program, implements the UI for running the feature selection algorithm(s)
    :return: null
    """
    data = util.parse(args.input)
    search_type = get_algorithm(args.algorithm)

    n = data[0][1].__len__()

    logging.info("This dataset has {} features (not including the class attribute), with {} instances."
                 .format(n, len(data)))

    logging.info("Please wait while I normalize the data... ")
    data = util.normalize(data)
    logging.info("Done!")

    accuracy = util.nearest_neighbor(data)
    logging.info('Running nearest neighbor with all {} features, using "leave-one-out" '
                 "evaluation, I get an accuracy of {}%".format(n, accuracy*100))

    logging.info("Beginning search.")

    start = timer()
    feature_set, accuracy = util.search(search_type, data)
    end = timer()

    logging.info("Finished search!! The best feature subset is {" + ', '.join(str(s+1) for s in feature_set) + "}, which has an accuracy of " + str(accuracy*100) + "%")
    logging.info("It took {} seconds to find this feature set.".format((end-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", choices={'ss', 'fs', 'bs'}, required=True,
                        help="ss = Tyson's special sauce, bs = backward selection, fs = forward selection")
    parser.add_argument('-i', '--input', required=True)

    args = parser.parse_args(sys.argv[1:])

    logging.basicConfig(level=logging.DEBUG)
    main(args)
