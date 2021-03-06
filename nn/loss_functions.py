import numpy as np
class SquaredError:
    """
    call returns (predictions - target)**2 / 2
    """

    def __call__(self, predictions, target):
        target = target.reshape(predictions.shape)
        return (predictions - target)**2 / 2
        #return np.divide(np.power(np.subtract(predictions, target), 2), 2)

    def prime(self, predictions, target):
        target = target.reshape(predictions.shape)
        return predictions - target
        #return np.subtract(predictions, target)

class CrossEntropy:
    """
    only available with a softmax output layer
    """

    def __init__(self, epsilon = 1e-11):
        self.epsilon = epsilon


    def __call__(self, predictions, target):
        """
        1e-9 is added to avoid np.log(0) -> divide by 0
        """
        N = predictions.shape[0]
        ce = -np.sum(target * np.log(predictions + 1e-9)) / N
        return ce

    def prime(self, predictions, target):
        return predictions - target