import numpy as np



class sigmoid:

    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def prime(self, z):
        return self(z) * (1 - self(z))

    def __repr__(self):
        return "Sigmoid"


class relu:

    def __call__(self, z):
        return np.maximum(0, z)

    def prime(self, z):
        return (z > 0).astype("float32")

    def __repr__(self):
        return "ReLu"


class lrelu:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, z):
        return np.maximum(self.alpha * z, z)

    def prime(self, z):
        return np.where(z <= 0, 0, self.alpha)

    def __repr__(self):
        return "Leaky ReLu"


class swish:

    def __call__(self, z):
        return z * sigmoid(z)

    def prime(self, z):
        return relu(z) + sigmoid(z)*(1 - relu(z))

    def __repr__(self):
        return "Swish"


class softmax:

    """
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

    def __call__(self, z):
        return z

    def prime(self, z):
        return np.ones(z.shape)

    def __repr__(self):
        return "Identity"


class tanh:

    def __call__(self, z):
        return np.tanh(z)

    def prime(self, z):
        return 1 - np.power(self(z),2)

    def __repr__(self):
        return "Tanh"