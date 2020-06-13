#!/usr/bin/env python3

from sensei.activations import LeakyRelu, Sigmoid
from sensei.network import Network

from sensei.optimizers import Adam
from sensei.costs import MSE

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test neural network library')

parser.add_argument('--epochs', dest='epochs', type=int, nargs=1,
                    default=[5000], help='Number of epochs (training cycles)')

parser.add_argument('--batchsize', dest='batch_size', type=int, nargs=1,
                    default=[1], help='Batch size of training samples')

args = parser.parse_args()

# all input vectors
X = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

# all output vectors
Y = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0]
    ])

# create network using the Adam optimizer
# and the MSE cost function
network = Network(Adam(X.shape[0]), MSE)

# set the network structure
network.add_layer(16, Sigmoid, X.shape[1])
network.add_layer(16, Sigmoid)
network.add_layer(Y.shape[1], LeakyRelu)

# train for a given amount of epochs
network.fit(X, Y, max(1, args.epochs[0]), max(1, args.batch_size[0]))

# print all predictions next to the expected values
for (x, y) in zip(X, Y):

    prediction = network.predict(x)
    print(f'{prediction}, {y}')
