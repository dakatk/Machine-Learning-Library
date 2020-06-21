#!/usr/bin/env python3

import sys

sys.path.append('..')

from sensei.activations import LeakyRelu, Sigmoid
from sensei.network import Network

from sensei.optimizers import Adam
from sensei.costs import MSE

import numpy as np
import unittest


class TestNetwork(unittest.TestCase):

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

    epochs = 5000
    batch_size = 1

    def __init__(self, *args, **kwargs):

        super(TestNetwork, self).__init__(*args, **kwargs)

        # create network using the Adam optimizer
        # and the MSE cost function
        self.network = Network(Adam(self.X.shape[0]), MSE)

        # TODO can we eliminate the need for explicit indexing of 'shape'?
        # set the network structure
        self.network.add_layer(16, Sigmoid, self.X.shape[1])
        self.network.add_layer(16, Sigmoid)
        self.network.add_layer(self.Y.shape[1], LeakyRelu)
        
    def test_network_setup(self):

        layers = self.network.layers

        # Check that all layers are accounted for
        self.assertEqual(len(layers), 3)

        # Check that each layer has the correct shape
        self.assertEqual(layers[0].weights.shape, (16, self.X.shape[1]))
        self.assertEqual(layers[1].weights.shape, (16, 16))
        self.assertEqual(layers[2].weights.shape, (self.Y.shape[1], 16))

    def test_network_fit(self):

        # Train the network
        self.network.fit(self.X, self.Y, self.epochs, self.batch_size)
        assertAlmost = np.vectorize(self.assertAlmostEqual)

        for (x, y) in zip(self.X, self.Y):

            prediction = self.network.predict(x)
            print(prediction, y, sep=' | ')

            # Check that the network's prediction is
            # within reasonable error of the actual data
            self.assertEqual(y, np.round(prediction))

    def test_network_save(self):

        # Redundant save/load test
        self.network.save('network.json')
        self.network = Network.load('network.json')

        layers = self.network.layers
        
        # Check that all layers are still accounted for
        self.assertEqual(len(layers), 3)

        # Check that each layer still has the correct shape
        self.assertEqual(layers[0].weights.shape, (16, self.X.shape[1]))
        self.assertEqual(layers[1].weights.shape, (16, 16))
        self.assertEqual(layers[2].weights.shape, (self.Y.shape[1], 16))


if __name__ == '__main__':
    unittest.main()
