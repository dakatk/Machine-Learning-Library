import numpy as np

## Cost (or loss) functions:

# 'np.vectorize' creates a function that applies itself to
# all elements any n-dimensional vector
class MAE(object):
    """ Mean absolute error (L1) loss function """

    _f_prime = np.vectorize(lambda i: -1.0 if i else 1.0)

    @staticmethod
    def prime(o, y):
        return MAE._f_prime(o < y)


class MSE(object):
    """ Mean squared error (L2) loss function """

    @staticmethod
    def prime(o, y):
        return o - y


class LogLoss(object):
    """ Binary crossentropy (logistic) loss function """

    @staticmethod
    def prime(o, y):

        inv_o = 1 - o
        inv_y = 1 - y

        # Avoid divide-by-zero errors:
        if np.any(o == 0.0):
            o = o + 1e-8

        if np.any(inv_o == 0.0):
            inv_o = inv_o + 1e-8

        return -((y / o) - (inv_y / inv_o))
