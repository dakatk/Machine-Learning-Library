import numpy as np


class _Optimizer(object):
    """ Template for each optimizer """

    def __init__(self, inputs, learning_rate):
        """
            inputs: the size of the inout set (NOT the size of an
                    individual input vector)
                    
            learning_rate: basically the step size of the gradient
                    descent algorithm. Smaller learning rate means
                    smoother steps, but longer training time
        """

        self.batch_indices = list(range(inputs))

        self.inputs = inputs
        self.learning_rate = learning_rate

    def next(self, t: int, batch_size: int) -> int:
        """ return the sample index for the current time step """
        pass

    def delta(self, layer_index: int, gradient: np.ndarray) -> np.ndarray:
        """ perform gradient descent to calculate the weight deltas
            to complete the backprop steps """
        pass

## Optimization functions:

class SGD(_Optimizer):
    """ Stochastic gradient descent optimizer """

    momentum = 0.9

    def __init__(self, inputs, *, learning_rate=0.01):

        # velocities will be populated when needed (defualts to zero
        # during first delta calculations)
        self.velocities = []

        super().__init__(inputs, learning_rate)

    def next(self, t, batch_size):
    
        np.random.shuffle(self.batch_indices)
        return self.batch_indices[:batch_size]

    def delta(self, layer_index, gradient):
    
        # default velocities corresponding to as given layer
        # to zero if they have not been previously populated
        if len(self.velocities) <= layer_index:
            self.velocities.append(np.zeros(gradient.shape))

        # classical momentum calculation:
        def moment(vel):

            nonlocal self, gradient
            return (vel * self.momentum) + (gradient * self.learning_rate)

        # applying the classical momentum equation allows the optimizer to
        # speed up along areas of low curvature along the gradient
        self.velocities[layer_index] = moment(self.velocities[layer_index])

        # applying classical momentum a second time creates something akin
        # to Nesterov acceleration, as postulated in the creation of the
        # Nadam optimizer function
        return moment(self.velocities[layer_index])


class Adam(_Optimizer):
    """ Adaptive momentum (or Adam) optimizer """

    beta1 = 0.9
    beta2 = 0.999

    batch_start = 0

    def __init__(self, inputs, *, learning_rate=0.002):

        # set when each sample is taken
        self.t = None

        # moments and velocities are populated during delta calculations
        self.moments = []
        self.velocities = []

        super().__init__(inputs, learning_rate)

    def next(self, t, batch_size):

        self.t = t

        batch = slice(self.batch_start, self.batch_start + batch_size)

        self.batch_start += batch_size
        
        if self.batch_start >= self.inputs:
            self.batch_start = 0

        # mini batch of size 'batch_size'
        return self.batch_indices[batch]

    def delta(self, layer_index, gradient):

        # populate moments and deltas (defaulting to 
        if len(self.moments) <= layer_index:
            self.moments.append(np.zeros(gradient.shape))

        if len(self.velocities) <= layer_index:
            self.velocities.append(np.zeros(gradient.shape))

        # momentum is applied similar to velocity in SGD
        self.moments[layer_index] = (self.beta1 * self.moments[layer_index]) + ((1 - self.beta1) * gradient)

        # velocity is applied using the squared gradient to fight against
        # the momentum getting as it gets too aggressive
        self.velocities[layer_index] = (self.beta2 * self.velocities[layer_index]) + ((1 - self.beta2) * (gradient ** 2))

        # momentum and velocity each become more prominent as the
        # time steps increase ('1 - (b^t)' approaches 1 as 't' increases for 'b < 1')
        moment_bar = self.moments[layer_index] / (1 - (self.beta1 ** self.t))
        velocity_bar = self.velocities[layer_index] / (1 - (self.beta2 ** self.t))

        velocity_sqr = np.sqrt(velocity_bar)

        # avoid divide-by-zero errors
        if (np.any(velocity_sqr == 0.0)):
            velocity_sqr += 1e-7

        # apply momentum in 'learning_rate' steps subdued by the calculated velocity
        # (square root is taken to mildly reduce the affect of velocity)
        return (self.learning_rate * moment_bar) / velocity_sqr


class AggMo(_Optimizer):
    """ Aggregate momentum optimizer """

    def __init__(self, inputs, *, k=3, learning_rate=0.01):

        # k = number of momentum calculations
        self.k = k

        # momentum constants (betas) calulated as an exponential set for k elements
        self.betas = [1 - (0.1 ** i) for i in range(k)]

        self.velocities = []

        super().__init__(inputs, learning_rate)

    def next(self, t):
        
        np.random.shuffle(self.batch_indices)
        return self.batch_indices[:batch_size]

    def delta(self, layer_index, gradient):

        # basically the same as SGD, except momentum is applied using
        # multiple momentum constants (betas) and the final delta is
        # calculated using the aggregate sum of the corresponding velocity set

        if len(self.velocities) <= layer_index:
            self.velocities.append([np.zeros(gradient.shape) for _ in range(self.k)])

        def moment(beta, vel):

            nonlocal gradient
            return beta * vel + gradient

        self.velocities[layer_index] = [moment(*args) for args in zip(self.betas, self.velocities[layer_index])]

        return (self.learning_rate * sum(self.velocities[layer_index])) / self.k
