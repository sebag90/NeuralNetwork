import numpy as np


class sigmoid:
    """
    takes a real value as input and outputs another value between 0 and 1.
    """

    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def prime(self, z):
        return self(z) * (1 - self(z))

    def __repr__(self):
        return "Sigmoid"


class relu:
    """
    returns 0 if the input z is lower than 0, otherwise is linear (returns z)
    """

    def __call__(self, z):
        return np.maximum(0, z)

    def prime(self, z):
        return (z > 0).astype("float32")

    def __repr__(self):
        return "ReLu"


class lrelu:
    """
    like relu, but instead of 0 it returns z multiplied by a (very small) alpha
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, z):
        return np.maximum(self.alpha * z, z)

    def prime(self, z):
        return np.where(z <= 0, 0, self.alpha)

    def __repr__(self):
        return "Leaky ReLu"


class swish:
    """
    new activation function proposed by google
    https://arxiv.org/abs/1710.05941
    """

    def __call__(self, z):
        return z * sigmoid(z)

    def prime(self, z):
        return relu(z) + sigmoid(z)*(1 - relu(z))

    def __repr__(self):
        return "Swish"


class softmax:

    """
    Softmax function calculates the probabilities distribution of 
    the event over ‘n’ different events (one output unit for every class) 

    ---
    the softmax activation function can only work as an output layer
    together with the cross entropy loss function.
    since the derivative is set to 1, the softmax layer will directly
    pass the combined derivative of softmax + cross entropy to the
    next layer
    """

    def __call__(self, z):
        m = np.max(z, axis=1, keepdims=True)
        e = np.exp(z - m)
        return e / np.sum(e, axis=1, keepdims=True)

    def prime(self, z):
        return 1

    def __repr__(self):
        return "Softmax"


class identity:
    """
    linear activation function (returns the input)
    """

    def __call__(self, z):
        return z

    def prime(self, z):
        return np.ones(z.shape)

    def __repr__(self):
        return "Identity"


class tanh:
    """
    Tanh squashes a real-valued number to the range [-1, 1]
    """

    def __call__(self, z):
        return np.tanh(z)

    def prime(self, z):
        return 1 - np.power(self(z),2)

    def __repr__(self):
        return "Tanh"