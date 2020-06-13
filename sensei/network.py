import numpy as np


class _Layer(object):
    """ Represents dense hidden layers and the output layer in a neural network """

    def __init__(self, neurons, inputs, activation_fn):

        self.weights = np.random.rand(neurons, inputs)
        self.bias = np.random.rand(neurons)

        # these values will be set later, correct data types
        # and vector shapes are guaranteed when set
        self.attached_layer = None
        self.activations = None
        
        self.deltas = []
        self.inputs = []

        self.activation_fn = activation_fn
        self.neurons = neurons

    def feedforward(self, input_vec):

        # input and activations should be 1D vectors
        self.inputs.append(input_vec)
        self.activations = np.dot(self.weights, input_vec) + self.bias

        return self.activation_fn.call(self.activations)

    def backprop(self, actual, target, cost_fn):

        # if no attached layer is found, then calculate output deltas
        if self.attached_layer is None:
            delta = cost_fn.prime(actual, target)

        # otherwise, calculate deltas between hidden layers
        else:
            delta = np.dot(self.attached_layer.weights.T, self.attached_layer.deltas[-1])

        # multiply result by derivative of the activation function
        delta *= self.activation_fn.prime(self.activations)
        self.deltas.append(delta)

    def update(self, index, optimizer):

        delta = self.deltas.pop()

        # gradient = input vector * delta for each delta
        # (creates a matrix with 'neurons' rows and 'input size' columns)
        gradient = np.dot(np.array([delta]).T, np.array([self.inputs.pop()]))

        # optimizer determines delta for the weights
        self.weights -= optimizer.delta(index, gradient)
        self.bias -= optimizer.learning_rate * delta


class Network(object):
    """ Represents a network of layers (input, hidden, and output) """

    def __init__(self, optimizer, cost_fn):

        # 'layers' array will be populated as the network structure is created
        self.layers = []

        self.optimizer = optimizer
        self.cost_fn = cost_fn

    def add_layer(self, neurons, activation_fn, input_size=None):
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

    def fit(self, inputs, outputs, epochs, batch_size):
        """ Train the network with a given input and output set for
            a maximum number of training cycles given by epochs """

        print('Training for', epochs, 'epochs\n')

        convergence = False

        for t in range(1, epochs + 1):

            # optimizer determines which sample is chosen at each step
            samples = self.optimizer.next(t, batch_size)

            # feed forward and to get the prediction of the current sample
            network_outputs = [self.predict(inputs[sample]) for sample in samples]

            # TODO enable batches during backpropo
    
            # backprop to calculate deltas between layers
            for layer in reversed(self.layers):
                
                for (output, sample) in zip(network_outputs, samples):
                    layer.backprop(output, outputs[sample], self.cost_fn)

            # update all layers after deltas have been calculated
            for (i, layer) in enumerate(self.layers):

                for _ in samples:
                    layer.update(i, self.optimizer)

            prediction = np.round(np.array([self.predict(i) for i in inputs]))

            # accuracy metric (check that all predictions can be reasonably
            # interpreted as the expected values)
            if np.all(prediction == outputs):

                if convergence:
                    break

                # if the accuracy metric shows convergence, training is done
                print('Convergence by accuracy at epoch', t, '\n')
                convergence = True
            
    def predict(self, input_vec):
        """ Given a single input vector, determine what the network's output will be """
 
        output = input_vec

        # feedforward through each layer
        for layer in self.layers:

            # previous layer's output becomes next layer's input 
            output = layer.feedforward(output)

        return output
