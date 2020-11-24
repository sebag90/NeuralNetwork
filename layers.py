import numpy as np



class Dense:

    def __init__(self, incoming_size, outgoing_size, activation):
        self.weight = np.random.uniform(-1, 1, (incoming_size, outgoing_size)).astype("float32")
        self.bias = np.random.uniform(-1, 1, (1, outgoing_size)).astype("float32")
        self.activation = activation
        self.memory_z = None
        self.incoming = None
        self.nabla_w = None
        self.nabla_b = None
        self.delta = None


    def __repr__(self):
        return f"Units: {self.weight.shape[1]},\tActivation: {self.activation},\tType: Dense"


    def forward(self, x):
        self.incoming = x
        z = np.dot(x, self.weight) + self.bias
        self.memory_z = z
       
        return self.activation(z)


    def backward(self, cost, learning_rate = 0.01):

        # calculate delta
        self.delta = cost * self.activation.prime(self.memory_z)

        # calculate nabla for weight and bias
        self.nabla_w = np.dot(self.delta.T, self.incoming).T
        self.nabla_b = np.expand_dims(self.delta.mean(axis=0), 0)

        to_return =  np.dot(self.delta, self.weight.T)

        # update weight and bias
        self.weight -= learning_rate * self.nabla_w
        self.bias   -= learning_rate * self.nabla_b

        return to_return