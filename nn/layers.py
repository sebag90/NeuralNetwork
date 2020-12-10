import numpy as np



class Dense:

    def __init__(self, incoming_size, outgoing_size, activation):
        """
        initialize a fully connected layer based on incoming
        and outgoing size of the matrix and activation function
        """
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
        """
        Forward step through the layer:
        input * weights + bias --> activation function
        z (input + weight + bias) will be saved for backpropagation

        input: matrix x
        output: matrix x
        """
        self.incoming = x
        z = np.add(np.matmul(x, self.weight), self.bias)
        self.memory_z = z
       
        return self.activation(z)


    def backward(self, cost, learning_rate = 0.01):
        """
        backpropagation and weight update through the layer
        the derivative of the cost with respect to the weights 
        is calculated and the weights are updated in a single 
        step (avoids second loop over layer to update)

        input: cost from next layer (right side)
        output:  cost for the previous layer (left side)
        """

        # calculate delta
        self.delta = np.multiply(cost, self.activation.prime(self.memory_z))
        
        # calculate nabla for weight and bias
        self.nabla_w = np.matmul(self.incoming.T, self.delta)
        self.nabla_b = np.sum(self.delta, axis=0, keepdims=True)     
        to_return =  np.matmul(self.delta, self.weight.T)

        # update weight and bias
        self.weight -= np.multiply(learning_rate, self.nabla_w)
        self.bias   -= np.multiply(learning_rate, self.nabla_b)

        return to_return