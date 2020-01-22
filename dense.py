import mxnet
import numpy
import time
import random


def batch_iterator(features, labels, batch_size):
    indices = numpy.arange(len(features))
    random.shuffle(indices)
    for i in range(0, len(features), batch_size):
        batch_indices = numpy.array(indices[i: min(i+batch_size, len(features))])
        yield features[batch_indices], labels[batch_indices]



def accuracy(features, labels, network):
    acc = mxnet.metric.Accuracy()
    output = network(features).reshape(len(features), sudoku_size, out_size)
    #output = network(features).reshape(len(features), out_size, sudoku_size)
    #output = numpy.transpose(output, (0,2,1))
    preds = mxnet.nd.argmax(output, axis=2)
    acc.update(preds=preds, labels=labels)
    return acc.get()

def train_network(network, epochs, train_f, train_l, test_f, test_l):
    network.collect_params().initialize(mxnet.init.Uniform())
    trainer = mxnet.gluon.Trainer(network.collect_params(), 'sgd', {'learning_rate': 0.1})
    softmax = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
    for epoch in range(0, epochs):
        start = time.time()
        for (features, labels) in batch_iterator(train_f, train_l, 64):
            with mxnet.autograd.record():
                output = network(features).reshape(len(features), sudoku_size, out_size)
                #output = network(features).reshape(len(features), out_size, sudoku_size)
                #output = numpy.transpose(output, (0,2,1))
                loss = softmax(output, labels)
            loss.backward()
            trainer.step(features.shape[0])
        stop = time.time()
        print("Epoch", epoch, ": accuracy =", accuracy(test_f, test_l, network)[1], ", time =", stop-start)

sudoku_size = 81
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



train_network(mxnet.gluon.nn.Dense(sudoku_size*out_size), 60, train_features, train_labels, test_features, test_labels)