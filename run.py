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

network = Network(Adam(X.shape[0]), MSE)

network.add_layer(16, Sigmoid, X.shape[1])
network.add_layer(16, Sigmoid)
network.add_layer(Y.shape[1], LeakyRelu)

errors = network.fit(X, Y, Accuracy(1), 5000)

for (x, y) in zip(X, Y):

    prediction = network.predict(x)
    print(x, y, prediction, sep=' | ')

plt.title('Error/Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')

indices = list(range(1, len(errors) + 1))
errors = list(map(np.sum, errors))

plt.plot(indices, errors)
plt.show()
