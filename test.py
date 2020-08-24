from sensei.network import Network

from sensei.activations import *
from sensei.optimizers import *
from sensei.costs import *
from sensei.metrics import *

import numpy as np

inputs = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

outputs = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

network = Network(SGD(inputs.shape[0], learning_rate=0.1), MSE)
network.add_layer(8, Sigmoid, inputs.shape[1])
network.add_layer(outputs.shape[1], Sigmoid)
network.fit(inputs, outputs, Accuracy(1), 10000)

for (in_vec, out_vec) in zip(inputs, outputs):
    print(in_vec, out_vec, network.predict(in_vec))

