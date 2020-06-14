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

parser.add_argument('--batch-size', dest='batch_size', type=int, nargs=1,
                    default=[1], help='Batch size of training samples')

args = parser.parse_args()

epochs = max(1, args.epochs[0])
batch_size = max(1, args.batch_size[0])

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

# TODO can we eliminate the need for explicit indexing of 'shape'?
# set the network structure
network.add_layer(16, Sigmoid, X.shape[1])
network.add_layer(16, Sigmoid)
network.add_layer(Y.shape[1], LeakyRelu)

# train for a given amount of epochs
network.fit(X, Y, epochs, batch_size)

# test saving the network to JSON file
print('Saving to \'network.json\'...')
network.save('network.json')
print('Saved!\n')

# test loading the network from JSON file
print('Loading from \'network.json\'...')
network = Network.load('network.json')
print('Loaded!\n')

# print all predictions next to the expected values
for (x, y) in zip(X, Y):

    prediction = network.predict(x)
    print(f'{prediction}, {y}')
