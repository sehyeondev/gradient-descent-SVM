import sys
import numpy as np

# Preprocessing - make train and data sets.
# x contains feature which is numpy array type and label.
def make_set(x_list, y_list, num):
    assert(len(x_list) == len(y_list) == num)
    result = []
    for i in range(num):
        x = map(int, x_list[i].split(','))
        x.append(1)
        y = int(y_list[i])
        result.append((np.array(x), y))
    return result

# Train according to SVM algorithm.
def svm_train(C, eta, train_set, w):
    d_w = np.zeros(len(w))
    for i in range(len(train_set)):
        x = train_set[i]
        # If y*(x*w) >= 1 then 0 else -y*x
        if (x[1] * np.dot(x[0], w) < 1):
            d_w += (-1)*x[1]*x[0]
    d_w = w + C*d_w
    return w - eta*d_w

# Test trained model from SVM algorithm.
# For test_set, predict label, and compare with real label.
# Return accuracy of test_set
def svm_test(test_set, w):
    correct = 0
    for x in test_set:
        # If sign of w*x + b and y are same, correct.
        if (x[1] * np.dot(x[0], w) >= 0):
            correct += 1
    return correct / float(len(test_set))

# k-fold cross validation.
# Let each subsets be the test data, and use the others as training data.
# Lastly, compute average of each test data.
def k_fold(sets, C, eta, d):
    accuracy = []
    for i in range(len(sets)):
        test_set = sets[i]
        train_sets = sets[:i] + sets[(i+1):]
        # w is initialized with zero vector, and last element is b,
        # which is also initialized with zero.
        w = np.zeros(d+1)
        for train in train_sets:
            # Train 500 times for each train set.
            for _ in range(500):
                w = svm_train(C, eta, train, w)
        accuracy.append(svm_test(test_set, w))
    return np.mean(np.array(accuracy))

# Main function
if __name__ =="__main__":
    # Get input
    f = open(sys.argv[1], 'r')  # features.txt
    q = open(sys.argv[2], 'r')  # labels.txt
    features = f.readlines()
    labels = q.readlines()

    # Divide into 10 chunks of size 600
    set_size = 600
    set_num = 10
    dim = 122
    sets = []
    for i in range(set_num):
        feature = features[set_size*i : set_size*(i+1)]
        label = labels[set_size*i : set_size*(i+1)]
        sets.append(make_set(feature, label, set_size))

    # SVM algorithm parameters
    C = 0.5
    eta = 0.001

    #print svm_test(test_set, w)
    print (k_fold(sets, C, eta, dim))
    print (C)
    print (eta)