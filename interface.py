# interface.py implements the text UI
# format for text UI taken from project prompt
import util
from timeit import default_timer as timer
import argparse
import sys
import logging


def GetFile():
    """
    Get the file from the command line.
    (I don't know why I didn't use argsparse...)
    :return: File to be parsed.
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
    logging.info(search_type)
    return search_type


def get_algorithm(arg):
    if arg.lower() == 'ss':
        return 3
    elif arg.lower() == 'fs':
        return 1
    elif arg.lower() == 'bs':
        return 2


def main(args):
    """
    Main method of the program, this runs the program
    :return: null
    """
    data = util.parse(args.input)

    search_type = get_algorithm(args.algorithm)

    n = len(data[0][1])

    logging.info("This dataset has {} features (not including the class attribute), with {} instances".format(n, len(data)))

    logging.info("Please wait while I normalize the data... ")
    data = util.normalize(data)
    logging.info("Done!")

    accuracy = util.nearest_neighbor(data)
    logging.info("Running nearest neighbor with all " + str(n) + " features, using \"leave-one-out\" evaluation, I get an accuracy of " + str(accuracy*100) + "%")

    logging.info("Beginning search.")

    start = timer()
    feature_set, accuracy = util.search(search_type, data)
    end = timer()

    logging.info("Finished search!! The best feature subset is {" + ', '.join(str(s+1) for s in feature_set) + "}, which has an accuracy of " + str(accuracy*100) + "%")
    logging.info("It took " + str(end-start) + " seconds to find this feature set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm',
                        help="fs | bs | ss {fs=forward selection|bs=backward selection|ss=Tyson's special sauce}",
                        default='bs', choices={'bs', 'fs', 'ss'}, required=True)
    parser.add_argument('-i', '--input', help="Input data to be analyzed", required=True)

    logging.basicConfig(level=logging.DEBUG)

    main(parser.parse_args(sys.argv[1:]))

