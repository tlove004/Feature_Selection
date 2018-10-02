# util.py -- where the magic happens
# All functions developed by Tyson Loveless
# forward selection and backward elimination modeled off of lecture notes on those topics in CS205
import copy
import math
import random

DEBUG = False
USESTACK = True


# normalizes feature values ( for each feature x, normalized x' = (x-mean)/std_dev )
def normalize(_data):
    # looking at features only, ignore class labels
    features = [feature for cls in _data for feature in cls[1:]]
    mean = sum(sum(row) for row in features) / float((len(features) * len(features[0])))
    # variance = average of squared difference for each value
    variance = sum((x - mean) ** 2 for row in features for x in row)
    # standard dev is square root of variance
    std_dev = math.sqrt(variance)
    normalized = [[((x - mean) / std_dev) for x in row] for row in features]
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
    """
    Calculates the nearest neighbor for some set of data.
    :param data: Set of data that needs to be classified.
    :return: Accuracy of k-1 leave one out validation.
    """
    features = set([i for i, x in enumerate(data[0][1])])
    return leave_one_out_cross_validation(data, features)


def euclidean_distance(x, y):
    """
    Calculates the Euclidean distance between two points.
    :param x: x coordinate
    :param y: y coordinate
    :return: euclidean distance between x and y points.
    """
    distance = 0
    for i, j in zip(x, y):
        distance += (i - j) ** 2
    return math.sqrt(distance)


# train data using n-1 data points, predict data point left out, record accuracy
def leave_one_out_cross_validation(data, features, best=0):
    correct = 0
    incorrect = 0
    for i in range(0, len(data)):
        distances = []
        cls = data[i][0]
        cls_features = [data[i][1][k] for k in features]
        for j in range(0, len(data)):
            if i == j:
                continue
            distances.append([j, euclidean_distance(cls_features, [data[j][1][k] for k in features])])

        distances.sort(key=lambda tup: tup[1])

        if cls == data[distances[0][0]][0]:
            correct += 1
        else:
            incorrect += 1
            if len(data) - incorrect < best:
                return 0

    return correct / float(len(data))


def search(option, data):
    # number of features
    num_features = data[0][1].__len__()
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
    best_per_level = {x: 0 for x in range(1, num_features + 1)}

    if option is 1:
        # forward selection
        while True:
            for i in range(level, num_features + 1):
                if DEBUG:
                    print "On the " + str(i) + "th level of the tree"
                best_so_far_accuracy = 0
                best_feature_this_level = 0
                count = 0
                temp = []
                updated = False

                for k in range(0, num_features):
                    if k == used:
                        continue
                    if current_feature_set.union({k}) in checked:
                        continue
                    if k not in current_feature_set:
                        if DEBUG:
                            print " --Considering adding feature " + str(k)
                        accuracy = leave_one_out_cross_validation(data, current_feature_set.union({k}),
                                                                  best_per_level[i] * 100)
                        temp.append([k, accuracy])
                        print "    Using feature(s) {" + ', '.join(
                            str(s + 1) for s in current_feature_set.union({k})) + "} accuracy is " + str(
                            accuracy * 100) + "%"
                        total += 1

                        if accuracy >= best_so_far_accuracy:
                            if best_per_level[i] <= accuracy:
                                best_per_level[i] = accuracy
                                for j in range(0, count):
                                    stack.pop()
                                count = 0
                            if current_feature_set.union({k}) in checked:
                                continue
                            if USESTACK and best_so_far_accuracy == best_per_level[
                                i] and best_so_far_accuracy == accuracy and accuracy > this_accuracy:
                                stack.append(
                                    [current_feature_set.union({best_feature_this_level}), i, copy.deepcopy(accuracy)])
                                count += 1
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
                                    print "feature set {" + ', '.join(
                                        str(s + 1) for s in best_feature_set) + "} has accuracy " + str(
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

                if best_so_far_accuracy <= 0:
                    print "\nNo improvement on this path\n"
                    break
                if updated:
                    print "\nFeature set {" + ', '.join(
                        str(s + 1) for s in current_feature_set) + "} was best, accuracy is " + str(
                        best_so_far_accuracy * 100) + "%\n"

                for index, acc in temp:
                    if acc < best_so_far_accuracy:
                        checked.append(current_feature_set.difference({best_feature_this_level}).union({index}))

            if not stack:
                break
            # checked.append(copy.deepcopy(current_feature_set))
            current_feature_set, level, this_accuracy = stack.pop()
            print "(Checking a different path that tied at level " + str(level) + ")"
            print "\nFeature set {" + ', '.join(
                str(s + 1) for s in current_feature_set) + "} was best, accuracy is " + str(this_accuracy) + "\n"
            level += 1

        print "total number expanded: " + str(total)

        return best_feature_set, best_accuracy
    # backward elimination
    elif option is 2:
        current_feature_set = set(i for i in range(0, num_features))
        while True:
            for i in range(num_features + 1 - level, 0, -1):
                if DEBUG:
                    print "On the " + str(i) + "th level of the tree"
                best_so_far_accuracy = 0
                worst_feature_this_level = 0
                count = 0
                temp = []
                updated = False

                for k in range(0, num_features):
                    if k == used:
                        continue
                    if current_feature_set.difference({k}) in checked:
                        continue
                    if k in current_feature_set:
                        if DEBUG:
                            print " --Considering removing feature " + str(k)
                        accuracy = leave_one_out_cross_validation(data, current_feature_set.difference({k}),
                                                                  best=best_per_level[i] * 100)
                        temp.append([k, accuracy])
                        print "    Using feature(s) {" + ', '.join(
                            str(s + 1) for s in current_feature_set.difference({k})) + "} accuracy is " + str(
                            accuracy * 100) + "%"
                        total += 1

                        if accuracy >= best_so_far_accuracy:
                            if best_per_level[i] <= accuracy:
                                best_per_level[i] = accuracy
                                for j in range(0, count):
                                    stack.pop()
                                count = 0
                            if current_feature_set.difference({k}) in checked:
                                continue
                            if USESTACK and best_per_level[
                                i] == accuracy and best_so_far_accuracy == accuracy and accuracy > this_accuracy:
                                stack.append([current_feature_set.difference({worst_feature_this_level}), num_features + 1 - i,
                                              copy.deepcopy(accuracy)])
                                count += 1
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
                                    print "feature set {" + ', '.join(
                                        str(s + 1) for s in best_feature_set) + "} has accuracy " + str(
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
                if best_so_far_accuracy <= 0:
                    print "\nNo improvement this path\n"
                    break
                if updated:
                    print "\nFeature set {" + ', '.join(
                        str(s + 1) for s in current_feature_set) + "} was best, accuracy is " + str(
                        best_so_far_accuracy * 100) + "%\n"

            if not stack:
                break
            current_feature_set, level, this_accuracy = stack.pop()
            print "(Checking a different path that tied at level " + str(num_features + 1 - level) + ")"
            print "\nFeature set {" + ', '.join(
                str(s + 1) for s in current_feature_set) + "} was best, accuracy is " + str(this_accuracy * 100) + "%\n"
            level += 1

        print "total number expanded: " + str(total)

        return best_feature_set, best_accuracy
    # my searching function

    # population = a set of feature sets (randomly generated at first)
    # selection = function to choose which features to keep
    # crossover = function to merge two feature sets together
    # mutation = function to randomly replace features with other features
    # fitness = leave_one_out_cross_validation (so as to compare directly with forward selection/backward elim)

    elif option is 3:
        # evaluations = 100
        best_feature_set = 0
        total = 0
        best_accuracy = 0
        print("Generating random population..."),
        population = init_population(num_features, data.__len__())
        print("Done!\n")
        while len(population) > num_features / 4:
            print("Finding most fit individuals...")
            population, fitness, best_feature_this_level, add = selection(data, population)
            total += add
            if len(population) == 0:
                break
            if fitness[0] > best_accuracy:
                best_accuracy = fitness[0]
                best_feature_set = copy.deepcopy(best_feature_this_level)
            print("\nPerforming crossover and mutations...")
            population = generation(population, fitness, num_features)
            print("Done!\n")
        print "Total number cross-validated: " + str(total)
        return best_feature_set, best_accuracy


def init_population(num_features, num_instances):
    population = set()
    length = num_features * int(math.sqrt(num_instances))
    while population.__len__() < length:
        feature_set = set()
        for i in range(0, random.randint(1, num_features - 1)):
            feature_set.add(random.randint(0, num_features - 1))
        population.add(frozenset(sorted(feature_set)))

    return population


def selection(data, population):
    fitness = []  # will hold accuracy for each feature
    total = 0
    for features in population:
        accuracy = leave_one_out_cross_validation(data, features)
        total += 1
        fitness.append([accuracy, features])
    fitness.sort(key=lambda tup: tup[0], reverse=True)  # sort by accuracy
    length = fitness.__len__() / 10
    best = [x[1] for i, x in enumerate(fitness) if i in range(0, length)]
    acc = [x[0] for i, x in enumerate(fitness) if i in range(0, length)]
    if length > 0:
        print "Top " + str(length) + " feature sets:\n{" + '\n{'.join(
            ', '.join(str(i + 1) for i in list(s)) + '} with accuracy ' + str(x) for s, x in zip(best, acc))
    return best, acc, fitness[0][1], total


def generation(population, fitness, num_features):
    pop = set()
    while len(pop) < len(population) * 5:
        for feature_set, fit in zip(population, fitness):
            # 25% crossover, 25% mutations, 50% deletions
            j = random.uniform(0, 1)
            if j > 0.75:
                new = mutation(feature_set, num_features)
            elif j < 0.50:
                new = delete(feature_set, population)
            else:
                new = crossover(feature_set, population)
            pop.add(feature_set)
            if len(new) == 0:
                continue
            if new not in pop:
                print "   New individual added to population: {" + ', '.join(str(s + 1) for s in new) + "}"
                pop.add(frozenset(new))
            else:
                new = mutation(feature_set, num_features)
                print "   New individual added to population: {" + ', '.join(str(s + 1) for s in new) + "}"
                pop.add(frozenset(new))
    new = best_crossover(population[0], population)
    pop.add(frozenset(new))
    return pop


def mutation(feature_set, num_features):
    features = [x for x in list(feature_set)]
    index = random.randint(0, len(features) - 1)
    change = random.randint(0, num_features - 1)
    features[index] = change
    return set(features)


#
def delete(feature_set, population):
    """
    Deletion chooses a feature to delete that is least used in population
    :param feature_set: feature set to iterate through
    :param population: population of given set
    :return: a new set of features with the least used feature removed.
    """
    features = [x for x in list(feature_set)]
    pop = [x for y in population for x in y]
    min = float("+inf")
    rem = features[0]
    for i in range(0, len(features)):
        x = pop.count(features[i])
        if x < min:
            min = x
            rem = features[i]
    features.remove(rem)
    return set(features)


def best_crossover(feature_set, population):
    """
    What is the best crossover for this feature?
    :param feature_set: set of features to be observed.
    :param population: population of given set.
    :return: a new set of features that include the best crossover.
    """
    new = []
    pop = [x for y in population for x in y]
    most = pop[0]
    max = float("-inf")
    all = list(set(pop))
    for j in range(0, len(feature_set)):
        for i in all:
            x = pop.count(i)
            if x > max:
                max = x
                most = i
        new.append(most)
        pop = filter(lambda a: a != most, pop)
        max = float("-inf")

    return set(new).union(feature_set)


def crossover(feature_set, population):
    features = [x for x in list(feature_set)]
    while True:
        mate = random.choice(tuple(population))
        if mate != features:
            break
    features2 = [x for x in list(mate)]
    return set(features).union(set(features2))
