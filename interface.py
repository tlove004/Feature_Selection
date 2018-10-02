# interface.py implements the text UI
# format for text UI taken from project prompt
import util
from timeit import default_timer as timer


def GetFile():
    """
    Get and parse data from the given file.
    :return: Parsed data.
    """
    print "Welcome to Tyson Loveless' Feature Selection Algorithm."
    name = raw_input('Type in the name of the file to test: ')
    return util.parse(name)


def GetAlgorithm():
    """
    Allows the user to select an algorithm to run on a dataset
    :return: algorithm to be used.
    """
    print "Type the number of the algorithm you want to run."
    print "\n   1)  Forward Selection"
    print "\n   2)  Backward Elimination"
    print "\n   3)  Tyson's Genetic Algorithm\n"
    search_type = input('                 ')
    return search_type


def main():
    """
    Main method of the program, implements the UI for running the feature selection algorithm(s)
    :return: null
    """
    data = GetFile()
    search_type = GetAlgorithm()

    n = data[0][1].__len__()

    print "\nThis dataset has " + str(n) + " features (not including the class attribute), with " \
          + str(data.__len__()) + " instances."

    print("\nPlease wait while I normalize the data... "),
    data = util.normalize(data)
    print "Done!"

    accuracy = util.nearest_neighbor(data)
    print "\nRunning nearest neighbor with all " + str(n) + " features, using \"leave-one-out\" evaluation, I get an accuracy of " + str(accuracy*100) + "%"

    print "\nBeginning search.\n"

    start = timer()
    feature_set, accuracy = util.search(search_type, data)
    end = timer()

    print "Finished search!! The best feature subset is {" + ', '.join(str(s+1) for s in feature_set) + "}, which has an accuracy of " + str(accuracy*100) + "%"
    print "\nIt took " + str(end-start) + " seconds to find this feature set."


main()
