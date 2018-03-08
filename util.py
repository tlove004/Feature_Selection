# utility functions
import math
import copy

DEBUG = False


# normalizes feature values ( for each feature x, normalized x' = (x-mean)/std_dev )
def normalize(_data):
    # looking at features only, ignore class labels
    features = [feature for cls in _data for feature in cls[1:]]
    mean = sum(sum(row) for row in features)/float((len(features)*len(features[0])))
    # variance = average of squared difference for each value
    variance = sum((x-mean)**2 for row in features for x in row)
    # standard dev is square root of variance
    std_dev = math.sqrt(variance)
    normalized = [[((x-mean)/std_dev) for x in row] for row in features]
    return [[_data[i][0], row] for i, row in enumerate(normalized)]


# entries are parsed to look like:
# data = [ [class, [feature1, feature2, feature3,...]], [class, [feature1, feature2, feature3,...]], ...]
def parse(file):
    data = []
    with open("205_proj2_data/" + file) as f:
        lines = [line.rstrip().split() for line in f]
    for line in lines:
        data.append([int(float(line[0])), [float(i) for i in line[1:]]])
    return data


def nearest_neighbor(data):
    features = set([i for i, x in enumerate(data[0][1])])
    return leave_one_out_cross_validation(data, features)


def euclidean_distance(x, y):
    distance = 0
    for i, j in zip(x, y):
        distance += (i-j)**2
    return math.sqrt(distance)


# train data using n-1 data points, predict data point left out, record accuracy
def leave_one_out_cross_validation(data, features, feature_to_add=None):
    correct = 0
    if feature_to_add is not None:
        _features = features.union({feature_to_add})
    else:
        _features = features
    for i in range(0, len(data)):
        distances = []
        cls = data[i][0]
        cls_features = [data[i][1][k] for k in _features]
        for j in range(0, len(data)):
            if i == j:
                continue
            distances.append([j, euclidean_distance(cls_features, [data[j][1][k] for k in _features])])

        distances.sort(key=lambda tup: tup[1])

        if cls == data[distances[0][0]][0]:
            correct += 1

    return correct/float(len(data))


def search(option, data):
    # number of features
    n = data[0][1].__len__()
    current_feature_set = set()
    best_accuracy = 0
    best_feature_set = set()
    maxima = False
    reset = False
    this_accuracy = 0
    stack = []
    level = 1
    used = -1
    total = 0
    checked = []

    if option is 1:
        # forward selection
        while True:
            for i in range(level, n+1):
                if DEBUG:
                    print "On the " + str(i) + "th level of the tree"
                best_so_far_accuracy = 0
                best_feature_this_level = 0
                count = 0
                temp = []
                updated = False
    
                for k in range (0, n):
                    if k == used:
                        continue
                    if current_feature_set.union({k}) in checked:
                        continue
                    if k not in current_feature_set:
                        if DEBUG:
                            print " --Considering adding feature " + str(k)
                        accuracy = leave_one_out_cross_validation(data, current_feature_set, k)
                        temp.append([k, accuracy])
                        print "    Using feature(s) {" + ', '.join(str(s+1) for s in current_feature_set.union({k})) + "} accuracy is " + str(accuracy*100) + "%"
                        total+=1
    
                        if accuracy >= best_so_far_accuracy:
                            if current_feature_set.union({k}) in checked:
                                continue
                            if best_so_far_accuracy == accuracy and accuracy > this_accuracy:
                                stack.append([current_feature_set.union({best_feature_this_level}), i, copy.deepcopy(accuracy)])
                                count+=1
                            else:
                                for j in range(0, count):
                                    stack.pop()
                                count = 0
                            best_so_far_accuracy = accuracy
                            updated = True
                            best_feature_this_level = k
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_feature_set = set(current_feature_set.union({k}))
                                if maxima:
                                    reset = True
                                if DEBUG:
                                    print "feature set {" + ', '.join(str(s+1) for s in best_feature_set) + "} has accuracy " + str(
                                    accuracy)

                if stack:
                    checked.append(current_feature_set.union({best_feature_this_level}))
                current_feature_set.add(best_feature_this_level)
                if not maxima and (best_so_far_accuracy < best_accuracy):
                    print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)"),
                    maxima = True
                if reset:
                    print("\n(Accuracy has increased, we have escaped a local maxima!)"),
                    maxima = False
                    reset = False
                if updated:
                    print "\nFeature set {" + ', '.join(str(s+1) for s in current_feature_set) + "} was best, accuracy is " + str(best_so_far_accuracy*100) + "%\n"
                
                for index, acc in temp:
                    if acc < best_so_far_accuracy:
                        checked.append(current_feature_set.difference({best_feature_this_level}).union({index}))
        
            if not stack:
                break
            #checked.append(copy.deepcopy(current_feature_set))
            current_feature_set, level, this_accuracy = stack.pop()
            print "(Checking a different path that tied at level " + str(level) + ")"
            print "\nFeature set {" + ', '.join(str(s+1) for s in current_feature_set) + "} was best, accuracy is " + str(this_accuracy) + "\n"
        
        # print "total number expanded: " + str(total)

        return best_feature_set, best_accuracy
    # backward elimination
    elif option is 2:
        current_feature_set = set(i for i in range(0, n))
        while True:
            for i in range(n-level, 0, -1):
                if DEBUG:
                    print "On the " + str(i) + "th level of the tree"
                best_so_far_accuracy = 0
                worst_feature_this_level = 0
                count = 0
                temp = []
                updated = False
    
                for k in range(0, n):
                    if k == used:
                        continue
                    if current_feature_set.difference({k}) in checked:
                        continue
                    if k in current_feature_set:
                        if DEBUG:
                            print " --Considering removing feature " + str(k)
                        accuracy = leave_one_out_cross_validation(data, current_feature_set.difference({k}))
                        temp.append([k, accuracy])
                        print "    Using feature(s) {" + ', '.join(str(s+1) for s in current_feature_set.difference({k})) + "} accuracy is " + str(accuracy*100) + "%"
                        total+=1
    
                        if accuracy >= best_so_far_accuracy:
                            if current_feature_set.difference({k}) in checked:
                                continue
                            if best_so_far_accuracy == accuracy and accuracy > this_accuracy:
                                stack.append([current_feature_set.difference({worst_feature_this_level}), i, copy.deepcopy(accuracy)])
                                count+=1
                            else:
                                for j in range(0, count):
                                    stack.pop()
                                count = 0
                            best_so_far_accuracy = accuracy
                            updated = True
                            worst_feature_this_level = k
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_feature_set = current_feature_set.difference({k})
                                if maxima:
                                    reset = True
                                if DEBUG:
                                    print "feature set {" + ', '.join(str(s+1) for s in best_feature_set) + "} has accuracy " + str(
                                    accuracy)
    
                if stack:
                    checked.append(current_feature_set.difference({worst_feature_this_level}))
                try:
                    current_feature_set.remove(worst_feature_this_level)
                except:
                    print(''),
                if not maxima and (best_so_far_accuracy < best_accuracy):
                    print("\n(Warning, Accuracy has decreased! Continuing search in case of local maxima)"),
                    maxima = True
                if reset:
                    print("\n(Accuracy has increased, we have escaped a local maxima!)"),
                    maxima = False
                    reset = False
                if updated:
                    print "\nFeature set {" + ', '.join(str(s+1) for s in current_feature_set) + "} was best, accuracy is " + str(
                        best_so_far_accuracy * 100) + "%\n"

            if not stack:
                break
            current_feature_set, level, this_accuracy = stack.pop()
            print "(Checking a different path that tied at level " + str(n-level) + ")"
            print "\nFeature set {" + ', '.join(
                str(s + 1) for s in current_feature_set) + "} was best, accuracy is " + str(this_accuracy*100) + "%\n"

        # print "total number expanded: " + str(total)

        return best_feature_set, best_accuracy
    # my searching function
    elif option is 3:
        while True:
            
            if False:
                break
        return best_feature_set, best_accuracy
