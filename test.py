from NN import Network
import numpy as np


np.random.seed(43)
net = Network()

net.init([3, 4, 10, 1], ["sigmoid", "sigmoid","sigmoid"], "cross_entropy")

x = np.random.rand(5, 3)
y = np.array([[1], [0], [0], [1], [1]])

y_result = net.predict(x)



nablas, deltas = net.backpropagation(x, y)

for i in nablas:
    print(nablas[i]["bias"])