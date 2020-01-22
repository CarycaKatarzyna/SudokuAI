import numpy
import sys

N = [100000, 10]
features = ['train_features.csv', 'test_features.csv']
labels = ['train_labels.csv', 'test_labels.csv']

f = open('sudoku.csv')

for i in range(0,2):
    g = [open(features[i], 'w'), open(labels[i], 'w')]
    for j in range(0, N[i]):
        line = f.readline()
        line_arr = numpy.array([i for i in line])
        line_arr = numpy.split(line_arr,2)
        g[0].write(', '.join(line_arr[0][:81]) + '\n')
        g[1].write(', '.join(line_arr[1][:81]) + '\n')
    g[0].close()
    g[1].close()
f.close()