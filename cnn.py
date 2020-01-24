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
Epoch 0 : accuracy = 0.23617022 , time = 104.82679677009583
Epoch 1 : accuracy = 0.41702127 , time = 104.63758301734924
Epoch 2 : accuracy = 0.5638298 , time = 104.64593720436096
Epoch 3 : accuracy = 0.65106386 , time = 104.53795099258423
Epoch 4 : accuracy = 0.6851064 , time = 104.73862981796265
Epoch 5 : accuracy = 0.7 , time = 105.30143117904663
"""

def batch_iterator(features, labels, batch_size):
    indices = numpy.arange(len(features))
    random.shuffle(indices)
    for i in range(0, len(features), batch_size):
        batch_indices = numpy.array(indices[i: min(i+batch_size, len(features))])
        yield features[batch_indices], labels[batch_indices]


def accuracy(features, labels, network):
    features = numpy.transpose(features, (0,2,1))
    features = features.reshape((len(features), in_size, sudoku_rows, sudoku_cols))
    zeros = features[:, 0, :, :] == 1
    zeros = zeros.reshape((len(features), sudoku_size))
    output = network(features).reshape(len(features), sudoku_size, out_size)
    preds = mxnet.nd.argmax(output, axis=2)
    equal = preds == labels
    summ = equal + zeros
    return (numpy.sum((equal + zeros)[:,:] == 2) / zeros.sum()).asnumpy()[0]


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
        print("Epoch", epoch, ": accuracy =", accuracy(test_f, test_l, network), ", time =", stop-start)

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


train_network(create_network(sudoku_size*out_size), 60, train_features, train_labels, test_features, test_labels)