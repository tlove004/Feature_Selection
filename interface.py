# interface.py implements the text UI
# format for text UI taken from project prompt
import util


def GetFile():
    print "Welcome to Tyson Loveless' Feature Selection Algorithm."
    name = raw_input('Type in the name of the file to test: ')
    return util.parse(name)
    #return util.parse("CS205_BIGtestdata__1.txt")


def GetAlgorithm():
    print "Type the number of the algorithm you want to run."
    print "\n   1)  Forward Selection"
    print "\n   2)  Backward Elimination"
    print "\n   3)  Tyson's Special Algorithm\n"
    search_type = input('                 ')
    return search_type


def main():
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

    feature_set, accuracy = util.search(search_type, data)

    print "Finished search!! The best feature subset is {" + ', '.join(str(s+1) for s in feature_set) + "}, which has an accuracy of " + str(accuracy*100) + "%"



main()
