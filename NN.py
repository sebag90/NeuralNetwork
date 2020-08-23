import numpy as np

class Network(object):


    # initialize network
    def __init__(self, sizes, activations):
        if len(sizes) -1 != len(activations):
            print("number of activations must be equal to number of layers - 1")
        else:
            self.num_layers = len(sizes)
            self.activations = activations
            self.sizes = sizes
            self.biases = [np.random.randn(y) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[1:], sizes[:-1])]
            self.activation_funcs = {
                "relu" : self.relu,
                "softmax" : self.softmax,
                "sigmoid" : self.sigmoid,
                "identity": self.identity
            }

    # activation functions and derivatives
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    #----------------------------------------

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        z[z > 0] = 1
        z[z <= 0] = 0
        return z

    #----------------------------------------

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    #----------------------------------------

    def identity(self, x):
        return x


    # feed forward
    def feedforward(self, a):
        for b, w, activation in zip(self.biases, self.weights, self.activations):
            activation = self.activation_funcs[activation]
            a = activation(np.dot(a, w)+b)
        return a
