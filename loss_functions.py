import numpy as np

class SquaredError:

    def __call__(self, predictions, target):
        target = target.reshape(predictions.shape)
        return (predictions - target)**2 / 2

    def prime(self, predictions, target):
        target = target.reshape(predictions.shape)
        return predictions - target


class CrossEntropy:
    """
    only available with a softmax output layer
    """

    def __init__(self, epsilon = 1e-11):
        self.epsilon = epsilon


    def __call__(self, predictions, target):
        N = predictions.shape[0]
        ce = -np.sum(target * np.log(predictions)) / N
        return ce

    def prime(self, predictions, target):
        return predictions - target