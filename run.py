#!/usr/bin/env python3

from sensei.network import Network

from sensei.activations import *
from sensei.optimizers import *
from sensei.costs import *
from sensei.metrics import *

from matplotlib import pyplot as plt

import numpy as np

X = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
])

Y = np.array([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])


def loss(e):
    return np.sum(e ** 2) / 2


# Adam test
network = Network(Adam(X.shape[0]), MSE)

network.add_layer(16, Sigmoid, X.shape[1])
network.add_layer(16, Sigmoid)
network.add_layer(Y.shape[1], LeakyRelu)
network.save('network.json')

errors = network.fit(X, Y, Accuracy(1), 5000)

indices = list(range(1, len(errors) + 1))
errors = list(map(loss, errors))

plt.plot(indices, errors)

# SGD test
network = Network.load('network.json')
network.optimizer = AggMo(X.shape[0])

errors = network.fit(X, Y, Accuracy(1), 5000)

indices = list(range(1, len(errors) + 1))
errors = list(map(loss, errors))

plt.plot(indices, errors)

# Show and compare plots
plt.title('Error/Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
