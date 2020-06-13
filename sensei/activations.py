import numpy as np

## Activation functions:

class Sigmoid(object):
    """ Logistic sigmoid function """

    @staticmethod
    def call(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def prime(x):
        
        s = Sigmoid.call(x)
        return s * (1 - s)


class Tanh(object):
    """ Hyperbolic tangent function """

    @staticmethod
    def call(x):
        return np.tanh(x)

    @staticmethod
    def prime(x):

        t = np.tanh(x)
        return 1 - (t ** 2)

# 'np.vectorize' creates a function that applies itself to
# all elements any n-dimensional vector
class LeakyRelu(object):
    """ Leaky rectified linear unit function """

    _f = np.vectorize(lambda i: i if i >= 0.0 else i * 0.01)
    _f_prime = np.vectorize(lambda i: 1.0 if i >= 0.0 else 0.01)

    @staticmethod
    def call(x):
        return LeakyRelu._f(x)

    @staticmethod
    def prime(x):
        return LeakyRelu._f_prime(x)


class Relu(object):
    """ Rectified linear unit function """

    _f = np.vectorize(lambda i: i if i > 0.0 else 0.0)

    @staticmethod
    def call(x):
        return Relu._f(x)

    @staticmethod
    def prime(x):
        return np.heaviside(x, 0.5)
