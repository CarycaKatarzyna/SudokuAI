import mxnet
import numpy
import time
import random

def create_network(outputs):
    net = mxnet.gluon.nn.Sequential()
    with net.name_scope():
        net.add(mxnet.gluon.nn.Conv2D(256, kernel_size=(3,3), activation='sigmoid'))
        net.add(mxnet.gluon.nn.BatchNorm())
        net.add(mxnet.gluon.nn.Flatten())
        net.add(mxnet.gluon.nn.Dense(units=outputs))
    return net

"""
Epoch 0 : accuracy = 0.18297872 , time = 105.27248620986938
Epoch 1 : accuracy = 0.29148936 , time = 104.39789748191833
Epoch 2 : accuracy = 0.50212765 , time = 104.84704399108887
Epoch 3 : accuracy = 0.6042553 , time = 104.68717002868652
Epoch 4 : accuracy = 0.63404256 , time = 110.57108283042908
Epoch 5 : accuracy = 0.70851064 , time = 104.72625756263733
"""

def batch_iterator(features, labels, batch_size):
    indices = numpy.arange(len(features))
    random.shuffle(indices)
    for i in range(0, len(features), batch_size):
        batch_indices = numpy.array(indices[i: min(i+batch_size, len(features))])
        yield features[batch_indices], labels[batch_indices]

def fill_one(features, prob):
    prob = numpy.insert(prob.asnumpy(), 81, numpy.full((9), -numpy.inf), axis=1)
    zeros = features[:, 0,:,:] == 1
    zeros = zeros.reshape((len(features), sudoku_size))    
    temp = numpy.tile(numpy.arange(start=1, stop=82), (len(features), 1))
    ids = (temp * zeros.asnumpy()) -1
    for i in range(0, len(features)):
        ids_row = ids[i]
        prob_row = prob[i]
        prob_row = prob_row[ids_row.astype(int)]
        fill_id = numpy.argmax(numpy.max(prob_row, axis=1), axis=0)
        fill_digit = numpy.argmax(prob_row[fill_id])
        features[i, 0, fill_id//sudoku_rows, fill_id%sudoku_cols] = 0
        features[i, fill_digit+1, fill_id//sudoku_rows, fill_id%sudoku_cols] = 1
    return features

def accuracy(features, labels, network, epoch):
    features = numpy.transpose(features, (0,2,1))
    features = features.reshape((len(features), in_size, sudoku_rows, sudoku_cols))
    zeros = features[:, 0, :, :] == 1
    zeros = zeros.reshape((len(features), sudoku_size))
    while count_solved(features) > 0:
        output = network(features).reshape(len(features), sudoku_size, out_size)
        features = fill_one(features, output)
    preds = mxnet.nd.argmax(features, axis=1).reshape(len(features), sudoku_size) - 1
    if epoch == 5:
        print(labels)
        print(preds)
    equal = preds == labels
    summ = equal + zeros
    return (numpy.sum((equal + zeros)[:,:] == 2) / zeros.sum()).asnumpy()[0]

def count_solved(features):
    zeros = features[:, 0, :, :] == 1
    count_zeros = zeros.sum(axis=2).sum(axis=1)
    count_zeros = count_zeros > 0
    return count_zeros.sum().asnumpy()[0]


def train_network(network, epochs, train_f, train_l, test_f, test_l):
    network.collect_params().initialize(mxnet.init.Uniform())
    trainer = mxnet.gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate': 0.1})
    softmax = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    for epoch in range(0, epochs):
        start = time.time()
        for (features, labels) in batch_iterator(train_f, train_l, 64):
            with mxnet.autograd.record():
                features = numpy.transpose(features, (0,2,1))
                features = features.reshape((len(features), in_size, sudoku_rows, sudoku_cols))
                output = network(features).reshape(len(features), sudoku_size, out_size)
                loss = softmax(output, labels)
            loss.backward()
            trainer.step(features.shape[0])
        stop = time.time()
        print("Epoch", epoch, ": accuracy =", accuracy(test_f, test_l, network, epoch), ", time =", stop-start)

sudoku_cols = 9
sudoku_rows = 9
sudoku_size = sudoku_rows * sudoku_cols
in_size = 10
out_size = 9

test_features = numpy.loadtxt("test_features.csv", delimiter=',')
test_labels = numpy.loadtxt("test_labels.csv", delimiter=',')
train_features = numpy.loadtxt("train_features.csv", delimiter=',')
train_labels = numpy.loadtxt("train_labels.csv", delimiter=',')


test_features = mxnet.ndarray.one_hot(mxnet.nd.array(test_features), in_size) 
test_labels = mxnet.nd.array(test_labels-1)
train_features = mxnet.ndarray.one_hot(mxnet.nd.array(train_features), in_size) 
train_labels = mxnet.nd.array(train_labels-1)

train_network(create_network(sudoku_size*out_size), 6, train_features, train_labels, test_features, test_labels)