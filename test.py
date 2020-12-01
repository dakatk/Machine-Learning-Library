from sensei.network import Network

from sensei.activations import *
from sensei.optimizers import *
from sensei.costs import *
from sensei.metrics import *

from matplotlib import pyplot as plt

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

network = Network(Adam(inputs.shape[0], learning_rate=0.05), MSE)
network.add_layer(8, Sigmoid, inputs.shape[1])
network.add_layer(outputs.shape[1], Sigmoid)
errors = network.fit(inputs, outputs, Delta(0.1), 10000)

network.save("network.json")

for (in_vec, out_vec) in zip(inputs, outputs):
    print(in_vec, out_vec, network.predict(in_vec))

iterations = [i + 1 for i in range(len(errors))]
errors = [np.sum(np.abs(e)) / len(e) for e in errors]

plt.plot(iterations, [0 for _ in iterations], 'r-')
plt.plot(iterations, errors, 'b.-')
plt.title('Error/Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
