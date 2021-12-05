import sys
import numpy as np

fname = sys.argv[1]
gname = sys.argv[2]
x = open(fname, "r")
y = open(gname, "r")
x_data = [list(map(float, line.split(','))) + [1.0] for line in x.readlines()]
y_data = list(map(float, y.readlines()))

# constants
epoch = 50
C = 0.1
eta = 0.001
data_size = 6000
tot_chunk = 10
chunk_size = data_size/tot_chunk
dim = 122
init_w = np.zeros(dim+1)

def svm_train(X_train, Y_train, w):
    df = np.copy(w)
    for i in range(len(X_train)):
        x = np.array(X_train[i])
        y = Y_train[i]
        if y*np.dot(x, w) < 1:
            df -= C*y*x
    w -= eta*df
    return w

# k-fold cross validation
all_acc = np.zeros(tot_chunk)
for i in range(tot_chunk):
    X_test = x_data[chunk_size*i : chunk_size*(i+1)]
    Y_test = y_data[chunk_size*i : chunk_size*(i+1)]
    X_train = x_data[:chunk_size*i] + x_data[chunk_size*(i+1):]
    Y_train = y_data[:chunk_size*i] + y_data[chunk_size*(i+1):]

    # train data
    fin_w = np.copy(init_w)
    for j in range(epoch):
        fin_w = svm_train(X_train, Y_train, fin_w)

    # test data and compute accuracy
    accuracy = np.sum(np.dot(X_test, fin_w)*Y_test > 0)/float(chunk_size)
    train_accuracy = np.sum(np.dot(X_train, fin_w)*Y_train > 0)/float(data_size-chunk_size)
    all_acc[i] = accuracy

# print result
avg_acc = np.mean(all_acc)
print(avg_acc)
print(C)
print(eta)