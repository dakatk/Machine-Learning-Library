import numpy as np
from .optimizers import _Optimizer


class _Layer(object):
    """ Represents dense hidden layers and the output layer in a neural network """

    def __init__(self, neurons: int, inputs: int, activation_fn: type):

        self.weights = np.random.rand(neurons, inputs)
        self.bias = np.random.rand(neurons)

        # these values will be set later, correct data types
        # and vector shapes are guaranteed when set
        self.attached_layer = None
        self.activations = None
        self.input_vec = None
        self.delta = None

        self.activation_fn = activation_fn
        self.neurons = neurons

    def feedforward(self, input_vec: np.ndarray) -> np.ndarray:

        # input and activations should be 1D vectors
        self.input_vec = input_vec
        self.activations = np.dot(self.weights, input_vec) + self.bias

        return self.activation_fn.call(self.activations)

    def backprop(self, actual: np.ndarray, target: np.ndarray, cost_fn: type) -> None:

        # if no attached layer is found, then calculate output deltas
        if self.attached_layer is None:
            self.delta = cost_fn.prime(actual, target)

        # otherwise, calculate deltas between hidden layers
        else:
            self.delta = np.dot(self.attached_layer.weights.T, self.attached_layer.delta)

        # multiply result by derivative of the activation function
        self.delta *= self.activation_fn.prime(self.activations)

    def update(self, index: int, optimizer: _Optimizer) -> None:

        # gradient = input vector * delta for each delta
        # (creates a matrix with 'neurons' rows and 'input size' columns)
        gradient = np.dot(np.array([self.delta]).T, np.array([self.input_vec]))

        # optimizer determines delta for the weights
        self.weights -= optimizer.delta(index, gradient)
        self.bias -= optimizer.learning_rate * self.delta


class Network(object):
    """ Represents a network of layers (input, hidden, and output) """

    def __init__(self, optimizer: _Optimizer, cost_fn: type):

        # 'layers' array will be populated as the network structure is created
        self.layers = []

        self.optimizer = optimizer
        self.cost_fn = cost_fn

    def add_layer(self, neurons: int, activation_fn: type, input_size: int = None) -> None:
        """
            Order of layers: input, hidden(s), output
            input_size should be an integer to define the input layer
            neurons should be the same value as the size of the output vectors to define the output layer
        """

        # First layer (input layer)
        if input_size is not None:
            layer = _Layer(neurons, input_size, activation_fn)

        # Hidden or output layer
        else:

            prev_layer = self.layers[-1]

            # size of the input to the next layer is the same as the number
            # of neurons in the previous layer (thanks to the result of the dot product)
            layer = _Layer(neurons, prev_layer.neurons, activation_fn)
            prev_layer.attached_layer = layer
            
        self.layers.append(layer)

    def fit(self, inputs: np.ndarray, outputs: np.ndarray, epochs: int, batch_size: int) -> None:
        """ Train the network with a given input and output set for
            a maximum number of training cycles given by epochs """

        print('Training for', epochs, 'epochs\n')

        convergence = False

        for t in range(1, epochs + 1):

            # optimizer determines which sample is chosen at each step
            samples = self.optimizer.next(t, batch_size)

            for sample in samples:

                # feed forward and to get the prediction of the current sample
                network_output = self.predict(inputs[sample])

                # backprop to calculate deltas between layers
                for layer in reversed(self.layers):
                    layer.backprop(network_output, outputs[sample], self.cost_fn)

                # update all layers after deltas have been calculated
                for (i, layer) in enumerate(self.layers):
                    layer.update(i, self.optimizer)

            predictions = np.round(np.array([self.predict(i) for i in inputs]))

            # accuracy metric (check that all predictions can be reasonably
            # interpreted as the expected values)
            if np.all(predictions == outputs):

                if convergence:
                    break

                # if the accuracy metric shows convergence, training is done
                print('Convergence by accuracy at epoch', t, '\n')
                convergence = True
            
    def predict(self, input_vec: np.ndarray) -> np.ndarray:
        """ Given a single input vector, determine what the network's output will be """
 
        output = input_vec

        # feedforward through each layer
        for layer in self.layers:

            # previous layer's output becomes next layer's input 
            output = layer.feedforward(output)

        return output
